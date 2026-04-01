"""
微调任务的 CRUD 操作。
"""

import json
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from app.core.enums import JobStatus
from app.db.models import FinetuneJob


def create_job(
    db: Session,
    job_id: str,
    status: str,
    request_data: dict,
    output_dir: str,
    log_path: str,
    max_steps: int,
) -> FinetuneJob:
    """
    创建新的微调任务记录。
    
    参数：
        db: 数据库会话
        job_id: 唯一任务标识符
        status: 任务状态（例如 "queued"）
        request_data: 请求参数字典
        output_dir: 输出目录路径
        log_path: 日志文件路径
        max_steps: 最大训练步数
    
    返回：
        创建的 FinetuneJob 实例
    """
    job = FinetuneJob(
        id=job_id,
        status=status,
        request_json=json.dumps(request_data),
        created_at=datetime.now(timezone.utc),
        output_dir=output_dir,
        log_path=log_path,
        max_steps=max_steps,
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
