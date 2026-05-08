"""模型推理服务。"""

from pathlib import Path
from typing import Any

import numpy as np

from app.schemas.request import InferCovGroup
from app.schemas.response import InferPredictionItem
from app.services.dataset_service import load_data
from app.services.model_service import load_local_model


class InferenceError(Exception):
    """推理业务异常。"""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _resolve_target_model_dir(release_model_dir: Path, target: str) -> Path:
    """按 target 解析子模型目录。"""
    candidate = release_model_dir / f"finetuned-ckpt_{target}"
    if candidate.exists() and candidate.is_dir():
        return candidate

    raise InferenceError(404, f"model for target '{target}' not found")


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
    cov_group: list[InferCovGroup],
    prediction_length: int,
    context_length: int,
    csv_path: str,
) -> list[InferPredictionItem]:
    """执行推理并返回按 cov_group 顺序的预测结果。"""
    release_model_dir = Path(model_path)
    if not release_model_dir.exists() or not release_model_dir.is_dir():
        raise InferenceError(404, "model path not found")

    csv_file = Path(csv_path)
    if not csv_file.exists() or not csv_file.is_file():
        raise InferenceError(404, "csv_path not found")

    required_columns: list[str] = []
    for group in cov_group:
        required_columns.append(group.target)
        required_columns.extend(group.covariates)
    required_columns = list(dict.fromkeys(required_columns))

    try:
        df = load_data(str(csv_file), target_columns=required_columns)
    except Exception as exc:
        raise InferenceError(400, f"invalid csv data: {exc}") from exc

    history_length = df.shape[0]
    if history_length <= context_length:
        raise InferenceError(400, "history length is insufficient")

    predictions: list[InferPredictionItem] = []
    for group in cov_group:
        target = group.target
        model_dir = _resolve_target_model_dir(release_model_dir, target)

        target_values = df[target].to_numpy(copy=True)
        cov_values = {
            col: df[col].to_numpy(copy=True) for col in group.covariates
        }
        expected_pred_len = history_length - context_length
        merged_prediction: list[float] = []

        try:
            pipeline = load_local_model(model_dir, device="cpu")
            start_idx = 0
            while len(merged_prediction) < expected_pred_len:
                end_idx = start_idx + context_length
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
                    prediction_length=prediction_length,
                )
                if not raw_outputs:
                    raise InferenceError(500, f"empty prediction for target '{target}'")

                chunk_prediction = _to_float_list(raw_outputs[0])
                if not chunk_prediction:
                    raise InferenceError(500, f"empty prediction for target '{target}'")

                remaining = expected_pred_len - len(merged_prediction)
                merged_prediction.extend(chunk_prediction[:remaining])
                start_idx += prediction_length
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError(500, f"inference failed for target '{target}'") from exc

        predictions.append(
            InferPredictionItem(
                target=target,
                prediction=merged_prediction,
            )
        )

    return predictions
