"""规范兼容接口的适配逻辑。"""

from datetime import datetime, timezone
from typing import Any

from fastapi import status
from fastapi.exceptions import RequestValidationError

from app.core.errors import ApiError
from app.schemas.response import JobDetailResponse
from app.schemas.standard_response import LossData, TrainJobStatusData

TERMINAL_STATUSES = {"completed", "failed", "cancelled", "earlyout"}


def map_status(raw_status: str) -> str:
    """将内部状态映射为规范状态。"""
    if raw_status == "queued":
        return "pending"
    return raw_status


def build_loss_data(metrics: dict[str, Any], last_loss: float | None) -> LossData:
    """将内部 metrics 结构适配成规范 loss_data。"""
    values: list[float] = []
    if metrics:
        first_series = next(iter(metrics.values()), [])
        if isinstance(first_series, list):
            values = [float(item) for item in first_series]
    steps = list(range(1, len(values) + 1))
    current_loss = float(last_loss) if last_loss is not None else (values[-1] if values else 0.0)
    return LossData(steps=steps, values=values, current_loss=current_loss)


def _to_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def calculate_duration_seconds(job: JobDetailResponse, now: datetime | None = None) -> int:
    """计算耗时秒数：运行中返回已耗时，完成后返回总耗时。"""
    current = now or datetime.now(timezone.utc)
    started_at = _to_utc(job.started_at) or _to_utc(job.created_at)
    finished_at = _to_utc(job.finished_at)
    if started_at is None:
        return 0
    end = finished_at or current
    duration = int((end - started_at).total_seconds())
    return duration if duration > 0 else 0


def adapt_job_detail(job: JobDetailResponse, now: datetime | None = None) -> TrainJobStatusData:
    """将现有任务详情适配成规范查询结构。"""
    status_value = map_status(job.status)
    is_completed = status_value in TERMINAL_STATUSES
    return TrainJobStatusData(
        job_id=job.job_id,
        is_completed=is_completed,
        status=status_value,
        loss_data=build_loss_data(job.metrics, job.progress.last_loss),
        duration=calculate_duration_seconds(job, now=now),
    )


def normalize_api_error(exc: Exception) -> ApiError:
    """将异常归一化为规范错误码。"""
    from fastapi import HTTPException
    from pydantic import ValidationError

    if isinstance(exc, ApiError):
        return exc

    if isinstance(exc, HTTPException):
        if exc.status_code == status.HTTP_404_NOT_FOUND:
            return ApiError(code=40401, message="job not found", http_status=404)
        if exc.status_code == status.HTTP_401_UNAUTHORIZED:
            return ApiError(code=40101, message="unauthorized", http_status=401)
        if exc.status_code in (status.HTTP_400_BAD_REQUEST, status.HTTP_422_UNPROCESSABLE_ENTITY):
            return ApiError(code=40001, message="invalid parameter", http_status=400)
        return ApiError(code=50001, message="internal error", http_status=500)

    if isinstance(exc, ValidationError):
        return ApiError(code=40001, message="invalid parameter", http_status=400)

    if isinstance(exc, RequestValidationError):
        return ApiError(code=40001, message="invalid parameter", http_status=400)

    return ApiError(code=50001, message="internal error", http_status=500)
