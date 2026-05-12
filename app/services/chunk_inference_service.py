"""分段推理服务。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from time import time
from typing import Any

import pandas as pd
from loguru import logger
from pandas.api.types import is_numeric_dtype

from app.core.config import get_settings
from app.schemas.request import InferCovGroup
from app.schemas.response import InferPredictionItem
from app.services.inference_service import (
    InferenceError,
    _resolve_inference_config,
    _resolve_target_model_dir,
    _to_float_list,
)
from app.services.model_metadata_service import ModelMetadataError, load_model_metadata
from app.services.model_service import load_local_model


@dataclass
class _TaskCacheEntry:
    model_path: str
    pipelines: dict[str, Any]
    cov_group: list[InferCovGroup]
    prediction_length: int
    context_length: int
    metadata_model_dir_map: dict[str, str]
    last_access_at: float


_TASK_CACHE: dict[str, _TaskCacheEntry] = {}
_TASK_CACHE_LOCK = RLock()


def _expire_stale_tasks_unlocked(now_ts: float) -> None:
    settings = get_settings()
    ttl = max(int(settings.chunk_infer_cache_ttl_seconds), 0)
    if ttl <= 0:
        return

    expired_task_ids = [
        task_id
        for task_id, entry in _TASK_CACHE.items()
        if now_ts - entry.last_access_at > ttl
    ]
    for task_id in expired_task_ids:
        _TASK_CACHE.pop(task_id, None)
        logger.info("chunk infer cache evicted by ttl: task_id={}", task_id)


def _ensure_capacity_unlocked() -> None:
    settings = get_settings()
    max_tasks = max(int(settings.chunk_infer_cache_max_tasks), 1)
    if len(_TASK_CACHE) < max_tasks:
        return
    raise InferenceError(500, "chunk inference cache is full")


def get_model_infer_config(model_path: str) -> tuple[int, int]:
    """读取模型默认推理配置。"""
    model_dir = Path(model_path)
    if not model_dir.exists() or not model_dir.is_dir():
        raise InferenceError(404, "model path not found")
    try:
        metadata = load_model_metadata(model_dir)
    except ModelMetadataError as exc:
        raise InferenceError(exc.code, exc.message) from exc
    return int(metadata.get("prediction_length")), int(metadata.get("context_length"))


def _load_task_entry(task_id: str, model_path: str) -> _TaskCacheEntry:
    model_dir = Path(model_path)
    if not model_dir.exists() or not model_dir.is_dir():
        raise InferenceError(404, "model path not found")

    cov_group, prediction_length, context_length, metadata_model_dir_map = _resolve_inference_config(
        release_model_dir=model_dir,
        cov_group=None,
        prediction_length=None,
        context_length=None,
        target_filter=None,
        require_metadata=True,
    )

    pipelines: dict[str, Any] = {}
    for group in cov_group:
        target_model_dir = _resolve_target_model_dir(model_dir, group, metadata_model_dir_map)
        pipelines[group.target] = load_local_model(target_model_dir, device="cpu")

    return _TaskCacheEntry(
        model_path=model_path,
        pipelines=pipelines,
        cov_group=cov_group,
        prediction_length=prediction_length,
        context_length=context_length,
        metadata_model_dir_map=metadata_model_dir_map,
        last_access_at=time(),
    )


def _get_or_create_task_entry(task_id: str, model_path: str) -> tuple[_TaskCacheEntry, bool]:
    with _TASK_CACHE_LOCK:
        now_ts = time()
        _expire_stale_tasks_unlocked(now_ts)

        entry = _TASK_CACHE.get(task_id)
        if entry is not None:
            if Path(entry.model_path).resolve() != Path(model_path).resolve():
                raise InferenceError(409, "task_id conflicts with another model_path")
            entry.last_access_at = now_ts
            logger.info("chunk infer cache hit: task_id={}, cache_size={}", task_id, len(_TASK_CACHE))
            return entry, True

        _ensure_capacity_unlocked()
        new_entry = _load_task_entry(task_id=task_id, model_path=model_path)
        _TASK_CACHE[task_id] = new_entry
        logger.info("chunk infer cache load: task_id={}, cache_size={}", task_id, len(_TASK_CACHE))
        return new_entry, False


def release_task_models(task_id: str) -> None:
    """释放任务缓存模型。"""
    with _TASK_CACHE_LOCK:
        removed = _TASK_CACHE.pop(task_id, None)
        if removed is not None:
            logger.info("chunk infer cache released: task_id={}, cache_size={}", task_id, len(_TASK_CACHE))


def _clear_task_cache_for_test() -> None:
    """测试辅助：清空缓存。"""
    with _TASK_CACHE_LOCK:
        _TASK_CACHE.clear()


def run_chunk_inference(
    task_id: str,
    model_path: str,
    segment: list[dict[str, object]],
    is_last_segment: bool,
) -> tuple[list[InferPredictionItem], bool]:
    """执行一次分段推理。"""
    entry, model_reused = _get_or_create_task_entry(task_id=task_id, model_path=model_path)
    try:
        df = pd.DataFrame(segment)
        for group in entry.cov_group:
            cols = [group.target, *group.covariates]
            for col in cols:
                if col not in df.columns:
                    raise InferenceError(400, f"missing column which model required: {col}")
                if not is_numeric_dtype(df[col]):
                    raise InferenceError(400, f"column '{col}' must be numeric")

        history_length = df.shape[0]
        if history_length <= entry.context_length:
            raise InferenceError(400, "history length is insufficient")

        predictions: list[InferPredictionItem] = []
        expected_pred_len = history_length - entry.context_length
        for group in entry.cov_group:
            target_values = df[group.target].to_numpy(copy=True)
            cov_values = {col: df[col].to_numpy(copy=True) for col in group.covariates}

            merged_prediction: list[float] = []
            start_idx = 0
            while len(merged_prediction) < expected_pred_len:
                end_idx = start_idx + entry.context_length
                input_dict: dict[str, Any] = {"target": target_values[start_idx:end_idx]}
                if cov_values:
                    input_dict["past_covariates"] = {
                        col: values[start_idx:end_idx] for col, values in cov_values.items()
                    }
                raw_outputs = entry.pipelines[group.target].predict(
                    inputs=[input_dict],
                    prediction_length=entry.prediction_length,
                )
                if not raw_outputs:
                    raise InferenceError(500, f"empty prediction for target '{group.target}'")
                chunk_prediction = _to_float_list(raw_outputs[0])
                if not chunk_prediction:
                    raise InferenceError(500, f"empty prediction for target '{group.target}'")
                remaining = expected_pred_len - len(merged_prediction)
                merged_prediction.extend(chunk_prediction[:remaining])
                start_idx += entry.prediction_length

            predictions.append(
                InferPredictionItem(
                    target=group.target,
                    prediction=merged_prediction,
                    actual=target_values[entry.context_length :].astype(float).tolist(),
                )
            )
        return predictions, model_reused
    finally:
        if is_last_segment:
            release_task_models(task_id)
