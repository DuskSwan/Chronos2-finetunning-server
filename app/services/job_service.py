"""业务逻辑服务层。

封装与任务相关的核心业务操作，如任务创建、状态更新等。
解耦数据库操作与 API 路由。
"""

from pathlib import Path
from typing import Any, Dict, Optional
import json
from datetime import datetime, timezone

from app.db.crud import (
    create_job,
    get_job_by_id,
    update_job_status,
    update_job_progress,
    mark_job_completed,
    mark_job_failed,
)
from app.core.enums import JobStatus
from app.core.config import Settings


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
    
    # 保存请求 JSON
    request_json_path = output_dir / "request.json"
    with open(request_json_path, "w", encoding="utf-8") as f:
        json.dump(request_data, f, indent=2, ensure_ascii=False)
    
    # 创建数据库记录
    db = SessionLocal()
    job = create_job(
        db=db,
        job_id=job_id,
        status=JobStatus.QUEUED.value,
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
        status=JobStatus.RUNNING.value,
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
