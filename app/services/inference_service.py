"""模型推理服务。"""

from pathlib import Path
from typing import Any

import numpy as np

from app.schemas.request import InferCovGroup
from app.schemas.response import InferPredictionItem
from app.services.dataset_service import load_data
from app.services.model_metadata_service import ModelMetadataError, load_model_metadata
from app.services.model_service import load_local_model


class InferenceError(Exception):
    """推理业务异常。"""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _resolve_target_model_dir(
    release_model_dir: Path,
    group: InferCovGroup,
    metadata_model_dir_map: dict[str, str] | None = None,
) -> Path:
    """按 group 解析子模型目录。"""
    model_dir = None
    if metadata_model_dir_map is not None:
        model_dir = metadata_model_dir_map.get(group.target)
    if isinstance(model_dir, str) and model_dir.strip():
        candidate = release_model_dir / model_dir.strip()
        if candidate.exists() and candidate.is_dir():
            return candidate

    target = group.target
    candidate = release_model_dir / f"finetuned-ckpt_{target}"
    if candidate.exists() and candidate.is_dir():
        return candidate
    raise InferenceError(404, f"model for target '{target}' not found")


def _to_infer_groups(payload: list[dict[str, Any]]) -> tuple[list[InferCovGroup], dict[str, str]]:
    groups: list[InferCovGroup] = []
    model_dir_map: dict[str, str] = {}
    for item in payload:
        group = InferCovGroup(
            target=item.get("target", ""),
            covariates=item.get("covariates") or [],
        )
        model_dir = item.get("model_dir")
        if isinstance(model_dir, str) and model_dir.strip():
            model_dir_map[group.target] = model_dir.strip()
        groups.append(group)
    return groups, model_dir_map


def _to_float_list(prediction: Any) -> list[float]:
    """将模型输出转换为 Python float 列表。"""
    if hasattr(prediction, "detach"):
        arr = prediction.detach().cpu().numpy()
    elif hasattr(prediction, "cpu") and hasattr(prediction, "numpy"):
        arr = prediction.cpu().numpy()
    else:
        arr = np.asarray(prediction)
    return np.asarray(arr).reshape(-1).astype(float).tolist()


def run_inference(
    model_path: str,
    cov_group: list[InferCovGroup] | None,
    prediction_length: int | None,
    context_length: int | None,
    csv_path: str,
) -> list[InferPredictionItem]:
    """执行推理并返回按 cov_group 顺序的预测结果。"""
    release_model_dir = Path(model_path)
    if not release_model_dir.exists() or not release_model_dir.is_dir():
        raise InferenceError(404, "model path not found")

    csv_file = Path(csv_path)
    if not csv_file.exists() or not csv_file.is_file():
        raise InferenceError(404, "csv_path not found")

    metadata: dict[str, Any] | None = None
    try:
        metadata = load_model_metadata(release_model_dir)
    except ModelMetadataError:
        metadata = None

    metadata_model_dir_map: dict[str, str] = {}
    final_cov_group = cov_group
    if final_cov_group is None and metadata is not None:
        final_cov_group, metadata_model_dir_map = _to_infer_groups(metadata.get("selected_groups") or [])
    if final_cov_group is None:
        raise InferenceError(400, "cov_group is required when metadata.json is missing")

    final_prediction_length = prediction_length
    if final_prediction_length is None and metadata is not None:
        final_prediction_length = metadata.get("prediction_length")
    if final_prediction_length is None:
        raise InferenceError(400, "prediction_length is required when metadata.json is missing")
    if final_prediction_length <= 0:
        raise InferenceError(400, "prediction_length must be a positive integer")

    final_context_length = context_length
    if final_context_length is None and metadata is not None:
        final_context_length = metadata.get("context_length")
    if final_context_length is None:
        raise InferenceError(400, "context_length is required when metadata.json is missing")
    if final_context_length <= 0:
        raise InferenceError(400, "context_length must be a positive integer")

    required_columns: list[str] = []
    for group in final_cov_group:
        required_columns.append(group.target)
        required_columns.extend(group.covariates)
    required_columns = list(dict.fromkeys(required_columns))

    try:
        df = load_data(str(csv_file), target_columns=required_columns)
    except Exception as exc:
        raise InferenceError(400, f"invalid csv data: {exc}") from exc

    for group in final_cov_group:
        cols = [group.target, *group.covariates]
        for col in cols:
            if col not in df.columns:
                raise InferenceError(400, f"missing column which model required: {col}")
            if not np.issubdtype(df[col].dtype, np.number):
                raise InferenceError(400, f"column '{col}' must be numeric")

    history_length = df.shape[0]
    if history_length <= final_context_length:
        raise InferenceError(400, "history length is insufficient")

    predictions: list[InferPredictionItem] = []
    for group in final_cov_group:
        target = group.target
        model_dir = _resolve_target_model_dir(release_model_dir, group, metadata_model_dir_map)

        target_values = df[target].to_numpy(copy=True)
        cov_values = {
            col: df[col].to_numpy(copy=True) for col in group.covariates
        }
        expected_pred_len = history_length - final_context_length
        merged_prediction: list[float] = []

        try:
            pipeline = load_local_model(model_dir, device="cpu")
            start_idx = 0
            while len(merged_prediction) < expected_pred_len:
                end_idx = start_idx + final_context_length
                input_dict: dict[str, Any] = {
                    "target": target_values[start_idx:end_idx],
                }
                if cov_values:
                    input_dict["past_covariates"] = {
                        col: values[start_idx:end_idx]
                        for col, values in cov_values.items()
                    }

                raw_outputs = pipeline.predict(
                    inputs=[input_dict],
                    prediction_length=final_prediction_length,
                )
                if not raw_outputs:
                    raise InferenceError(500, f"empty prediction for target '{target}'")

                chunk_prediction = _to_float_list(raw_outputs[0])
                if not chunk_prediction:
                    raise InferenceError(500, f"empty prediction for target '{target}'")

                remaining = expected_pred_len - len(merged_prediction)
                merged_prediction.extend(chunk_prediction[:remaining])
                start_idx += final_prediction_length
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError(500, f"inference failed for target '{target}'") from exc

        predictions.append(
            InferPredictionItem(
                target=target,
                prediction=merged_prediction,
                actual=target_values[final_context_length:].astype(float).tolist(),
            )
        )

    return predictions
