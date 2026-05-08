"""
微调任务的 CRUD 操作。
"""

import json
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from app.core.enums import JobStatus
from app.db.models import FinetuneJob, FinetuneJobLoss


def create_job(
    db: Session,
    job_id: str,
    status: str,
    request_json: str,
    output_dir: str,
    log_path: str,
    max_steps: int = 0,
) -> FinetuneJob:
    """
    创建新的微调任务记录。
    
    参数：
        db: 数据库会话
        job_id: 唯一任务标识符
        status: 任务状态（例如 "queued"）
        request_json: 请求参数 JSON 字符串
        output_dir: 输出目录路径
        log_path: 日志文件路径
        max_steps: 最大训练步数
    
    返回：
        创建的 FinetuneJob 实例
    """
    job = FinetuneJob(
        id=job_id,
        status=status,
        request_json=request_json,
        created_at=datetime.now(timezone.utc),
        output_dir=output_dir,
        log_path=log_path,
        max_steps=max_steps if max_steps > 0 else 0,
        current_step=0,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_job_by_id(db: Session, job_id: str) -> Optional[FinetuneJob]:
    """
    根据 ID 获取任务。
    
    参数：
        db: 数据库会话
        job_id: 任务标识符
    
    返回：
        找到则返回 FinetuneJob，否则返回 None
    """
    return db.query(FinetuneJob).filter(FinetuneJob.id == job_id).first()


def list_jobs_by_status(db: Session, status: str) -> list[FinetuneJob]:
    """按状态获取任务列表（按创建时间升序）。"""
    return (
        db.query(FinetuneJob)
        .filter(FinetuneJob.status == status)
        .order_by(FinetuneJob.created_at.asc())
        .all()
    )


def list_jobs(
    db: Session,
    skip: int = 0,
    limit: int = 100,
) -> list[FinetuneJob]:
    """
    列出所有任务并支持分页。
    
    参数：
        db: 数据库会话
        skip: 跳过的任务数
        limit: 返回的最大任务数
    
    返回：
        FinetuneJob 实例列表
    """
    return db.query(FinetuneJob).offset(skip).limit(limit).all()


def delete_job_by_id(db: Session, job_id: str) -> bool:
    """按 ID 删除任务。"""
    rows = db.query(FinetuneJob).filter(FinetuneJob.id == job_id).delete()
    db.commit()
    return rows > 0


def list_recent_jobs(
    db: Session,
    limit: int = 20,
) -> list[FinetuneJob]:
    """获取最近创建的任务列表。

    参数：
        db: 数据库会话
        limit: 返回的最大任务数

    返回：
        FinetuneJob 实例列表（按创建时间倒序）
    """
    return (
        db.query(FinetuneJob)
        .order_by(FinetuneJob.created_at.desc())
        .limit(limit)
        .all()
    )


def list_recent_jobs_by_status(
    db: Session,
    status: str,
    limit: int = 20,
) -> list[FinetuneJob]:
    """按状态获取最近创建的任务列表（按创建时间倒序）。"""
    return (
        db.query(FinetuneJob)
        .filter(FinetuneJob.status == status)
        .order_by(FinetuneJob.created_at.desc())
        .limit(limit)
        .all()
    )


def list_queued_jobs(db: Session) -> list[FinetuneJob]:
    """按创建时间升序获取处于 queued 状态的任务。"""
    return (
        db.query(FinetuneJob)
        .filter(FinetuneJob.status == JobStatus.queued.value)
        .order_by(FinetuneJob.created_at.asc())
        .all()
    )


def list_running_jobs(db: Session) -> list[FinetuneJob]:
    """按创建时间升序获取处于 running 状态的任务。"""
    return (
        db.query(FinetuneJob)
        .filter(FinetuneJob.status == JobStatus.running.value)
        .order_by(FinetuneJob.created_at.asc())
        .all()
    )


def update_job_status(
    db: Session,
    job_id: str,
    status: str,
    started_at: Optional[datetime] = None,
) -> Optional[FinetuneJob]:
    """
    更新任务状态。
    
    参数：
        db: 数据库会话
        job_id: 任务 ID
        status: 新状态
        started_at: 开始时间（如果适用）
    
    返回：
        更新后的 FinetuneJob，若不存在返回 None
    """
    job = get_job_by_id(db, job_id)
    if not job:
        return None
    
    job.status = status
    if started_at:
        job.started_at = started_at
    
    db.commit()
    db.refresh(job)
    return job


def update_job_progress(
    db: Session,
    job_id: str,
    current_step: int,
    max_steps: Optional[int] = None,
    last_loss: Optional[float] = None,
) -> Optional[FinetuneJob]:
    """
    更新任务训练进度。
    
    参数：
        db: 数据库会话
        job_id: 任务 ID
        current_step: 当前步数
        max_steps: 总步数（可选）
        last_loss: 最新损失值
    
    返回：
        更新后的 FinetuneJob，若不存在返回 None
    """
    job = get_job_by_id(db, job_id)
    if not job:
        return None
    
    job.current_step = current_step
    if max_steps is not None:
        job.max_steps = max_steps
    if last_loss is not None:
        job.last_loss = last_loss
    
    db.commit()
    db.refresh(job)
    return job


def _serialize_model_paths(model_paths: list[str]) -> str:
    return json.dumps(model_paths, ensure_ascii=False)


def _serialize_target_model_map(target_model_map: dict[str, str]) -> str:
    return json.dumps(target_model_map, ensure_ascii=False)


def mark_job_completed(
    db: Session,
    job_id: str,
    target_model_map: Optional[dict[str, str]] = None,
    model_paths: Optional[list[str]] = None,
    finished_at: Optional[datetime] = None,
) -> Optional[FinetuneJob]:
    """
    标记任务为完成。
    
    参数：
        db: 数据库会话
        job_id: 任务 ID
        target_model_map: target 到模型目录的映射
        model_paths: 兼容字段，旧版模型路径列表
        finished_at: 完成时间
    
    返回：
        更新后的 FinetuneJob，若不存在返回 None
    """
    job = get_job_by_id(db, job_id)
    if not job:
        return None
    
    if target_model_map is None:
        target_model_map = {}
    if model_paths is None:
        model_paths = list(target_model_map.values())
    if not target_model_map and model_paths:
        target_model_map = {
            f"group_{idx + 1}": str(path)
            for idx, path in enumerate(model_paths)
        }

    job.status = JobStatus.completed.value
    job.target_model_map = _serialize_target_model_map(target_model_map)
    job.model_paths = _serialize_model_paths(model_paths)
    job.finished_at = finished_at or datetime.now(timezone.utc)
    
    db.commit()
    db.refresh(job)
    return job


def mark_job_failed(
    db: Session,
    job_id: str,
    error_message: str,
    finished_at: Optional[datetime] = None,
) -> Optional[FinetuneJob]:
    """
    标记任务为失败。
    
    参数：
        db: 数据库会话
        job_id: 任务 ID
        error_message: 错误信息
        finished_at: 完成时间
    
    返回：
        更新后的 FinetuneJob，若不存在返回 None
    """
    job = get_job_by_id(db, job_id)
    if not job:
        return None
    
    job.status = JobStatus.failed.value
    job.error_message = error_message
    job.finished_at = finished_at or datetime.now(timezone.utc)
    
    db.commit()
    db.refresh(job)
    return job


def set_cancel_requested(
    db: Session,
    job_id: str,
    cancel_requested: bool = True,
) -> Optional[FinetuneJob]:
    """设置任务取消请求标记。

    参数：
        db: 数据库会话
        job_id: 任务 ID
        cancel_requested: 是否请求取消

    返回：
        更新后的 FinetuneJob，若不存在返回 None
    """
    job = get_job_by_id(db, job_id)
    if not job:
        return None

    job.cancel_requested = cancel_requested
    db.commit()
    db.refresh(job)
    return job


def mark_job_cancelled(
    db: Session,
    job_id: str,
    finished_at: Optional[datetime] = None,
) -> Optional[FinetuneJob]:
    """标记任务为取消。

    参数：
        db: 数据库会话
        job_id: 任务 ID
        finished_at: 完成时间

    返回：
        更新后的 FinetuneJob，若不存在返回 None
    """
    job = get_job_by_id(db, job_id)
    if not job:
        return None

    job.status = JobStatus.cancelled.value
    job.cancel_requested = True
    job.finished_at = finished_at or datetime.now(timezone.utc)
    db.commit()
    db.refresh(job)
    return job


def upsert_job_loss_point(
    db: Session,
    job_id: str,
    group_index: int,
    step: int,
    loss: float,
    target: Optional[str] = None,
) -> FinetuneJobLoss:
    """插入或更新任务的 loss 曲线点。"""
    normalized_target = (target or "").strip() or f"group_{group_index + 1}"
    point = (
        db.query(FinetuneJobLoss)
        .filter(
            FinetuneJobLoss.job_id == job_id,
            FinetuneJobLoss.target == normalized_target,
            FinetuneJobLoss.step == step,
        )
        .first()
    )

    if point is None:
        point = FinetuneJobLoss(
            job_id=job_id,
            group_index=group_index,
            target=normalized_target,
            step=step,
            loss=loss,
            created_at=datetime.now(timezone.utc),
        )
        db.add(point)
    else:
        point.group_index = group_index
        point.target = normalized_target
        point.loss = loss

    db.commit()
    db.refresh(point)
    return point


def list_job_loss_points(
    db: Session,
    job_id: str,
) -> list[FinetuneJobLoss]:
    """按 step 升序获取任务的 loss 曲线点。"""
    return (
        db.query(FinetuneJobLoss)
        .filter(FinetuneJobLoss.job_id == job_id)
        .order_by(FinetuneJobLoss.target.asc(), FinetuneJobLoss.step.asc())
        .all()
    )
