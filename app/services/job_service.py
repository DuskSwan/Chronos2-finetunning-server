"""业务逻辑服务层。

封装与任务相关的核心业务操作，如任务创建、状态更新等。
解耦数据库操作与 API 路由。
"""

from pathlib import Path
from typing import Any, Dict, Optional
import json
from datetime import datetime, timezone

from fastapi import HTTPException, status

from app.db.crud import (
    create_job,
    get_job_by_id,
    update_job_status,
    update_job_progress,
    mark_job_completed,
    mark_job_failed,
    list_recent_jobs,
    set_cancel_requested,
    mark_job_cancelled,
)
from app.core.enums import JobStatus
from app.core.config import Settings
from app.schemas.response import (
    JobDetailResponse,
    JobProgressResponse,
    JobResultResponse,
    JobListItemResponse,
    JobListResponse,
    CancelJobResponse,
)


def create_finetune_job(
    request_data: Dict[str, Any],
    settings: Settings,
) -> str:
    """创建微调任务并保存请求信息。

    Args:
        request_data: 创建任务的请求数据。
        settings: 应用配置对象。

    Returns:
        生成的 job_id。
    """
    import uuid
    from app.db.session import SessionLocal
    
    # 生成 job_id
    job_id = str(uuid.uuid4())
    
    # 确定输出目录
    output_root = Path(request_data.get("output_root") or settings.artifacts_root)
    output_dir = output_root / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取日志路径
    logs_dir = Path(settings.logs_root) / job_id
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(logs_dir / "train.log")
    
    # 保存请求 JSON（可配置关闭）
    if settings.save_request_artifacts:
        request_json_path = output_dir / "request.json"
        with open(request_json_path, "w", encoding="utf-8") as f:
            json.dump(request_data, f, indent=2, ensure_ascii=False)
    
    # 创建数据库记录
    db = SessionLocal()
    job = create_job(
        db=db,
        job_id=job_id,
        status=JobStatus.queued.value,
        request_json=json.dumps(request_data, ensure_ascii=False),
        output_dir=str(output_dir),
        log_path=log_path,
    )
    db.close()
    
    return job_id


def start_job_training(db: Any, job_id: str) -> None:
    """标记任务为运行状态。

    Args:
        db: 数据库会话。
        job_id: 任务 ID。
    """
    update_job_status(
        db=db,
        job_id=job_id,
        status=JobStatus.running.value,
        started_at=datetime.now(timezone.utc),
    )


def update_job_step(
    db: Any,
    job_id: str,
    current_step: int,
    max_steps: int,
    last_loss: Optional[float] = None,
) -> None:
    """更新任务训练进度。

    Args:
        db: 数据库会话。
        job_id: 任务 ID。
        current_step: 当前步数。
        max_steps: 总步数。
        last_loss: 最新损失值。
    """
    update_job_progress(
        db=db,
        job_id=job_id,
        current_step=current_step,
        max_steps=max_steps,
        last_loss=last_loss,
    )


def complete_job_training(
    db: Any,
    job_id: str,
    model_path: str,
) -> None:
    """标记任务为完成状态。

    Args:
        db: 数据库会话。
        job_id: 任务 ID。
        model_path: 微调模型路径。
    """
    mark_job_completed(
        db=db,
        job_id=job_id,
        model_path=model_path,
        finished_at=datetime.now(timezone.utc),
    )


def fail_job_training(
    db: Any,
    job_id: str,
    error_message: str,
) -> None:
    """标记任务为失败状态。

    Args:
        db: 数据库会话。
        job_id: 任务 ID。
        error_message: 错误信息。
    """
    mark_job_failed(
        db=db,
        job_id=job_id,
        error_message=error_message,
        finished_at=datetime.now(timezone.utc),
    )


def get_job_detail(db: Any, job_id: str) -> JobDetailResponse:
    """获取任务详情。

    Args:
        db: 数据库会话。
        job_id: 任务 ID。

    Returns:
        任务详情响应。
    """
    job = get_job_by_id(db, job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"任务不存在: {job_id}",
        )

    progress = JobProgressResponse(
        current_step=job.current_step,
        max_steps=job.max_steps,
        last_loss=job.last_loss,
    )

    return JobDetailResponse(
        job_id=job.id,
        status=job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        progress=progress,
        error_message=job.error_message,
        log_path=job.log_path,
        model_path=job.model_path,
    )


def get_job_result(db: Any, job_id: str) -> JobResultResponse:
    """获取任务结果。

    Args:
        db: 数据库会话。
        job_id: 任务 ID。

    Returns:
        任务结果响应。
    """
    job = get_job_by_id(db, job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"任务不存在: {job_id}",
        )

    if job.status != JobStatus.completed.value:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"任务未完成，当前状态: {job.status}",
        )

    return JobResultResponse(
        job_id=job.id,
        status=job.status,
        output_dir=job.output_dir,
        model_path=job.model_path,
        metrics={},
    )


def read_job_log(db: Any, job_id: str, tail: Optional[int] = None) -> str:
    """读取任务日志。

    Args:
        db: 数据库会话。
        job_id: 任务 ID。
        tail: 仅返回最后 N 行（可选）。

    Returns:
        日志文本。
    """
    job = get_job_by_id(db, job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"任务不存在: {job_id}",
        )

    if not job.log_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="日志路径不存在",
        )

    log_path = Path(job.log_path)
    if not log_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="日志文件不存在",
        )

    if tail is not None and tail <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="tail 必须是正整数",
        )

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.read().splitlines()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"读取日志失败: {exc}",
        ) from exc

    if tail is not None:
        lines = lines[-tail:]

    return "\n".join(lines)


def list_job_summaries(db: Any, limit: int = 20) -> JobListResponse:
    """获取任务列表摘要。

    Args:
        db: 数据库会话。
        limit: 返回的最大条数。

    Returns:
        任务列表响应。
    """
    if limit <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="limit 必须是正整数",
        )

    jobs = list_recent_jobs(db, limit=limit)
    items = [
        JobListItemResponse(
            job_id=job.id,
            status=job.status,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
        )
        for job in jobs
    ]

    return JobListResponse(items=items)


def request_cancel_job(db: Any, job_id: str) -> CancelJobResponse:
    """请求取消任务。

    Args:
        db: 数据库会话。
        job_id: 任务 ID。

    Returns:
        取消任务响应。
    """
    job = get_job_by_id(db, job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"任务不存在: {job_id}",
        )

    if job.status == JobStatus.queued.value:
        cancelled_job = mark_job_cancelled(db, job_id)
        return CancelJobResponse(
            job_id=cancelled_job.id,
            status=cancelled_job.status,
            cancel_requested=cancelled_job.cancel_requested,
            message="任务已取消（queued）",
        )

    if job.status == JobStatus.running.value:
        updated_job = set_cancel_requested(db, job_id, True)
        return CancelJobResponse(
            job_id=updated_job.id,
            status=updated_job.status,
            cancel_requested=updated_job.cancel_requested,
            message="已请求取消，等待训练停止",
        )

    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail=f"当前状态无法取消: {job.status}",
    )
