"""
微调任务的 SQLAlchemy 模型。
"""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, Float, Integer, String, Boolean, Text
from sqlalchemy.orm import Mapped, declarative_base, mapped_column

Base = declarative_base()


class FinetuneJob(Base):
    """微调任务模型。"""
    
    __tablename__ = "finetune_jobs"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="queued")
    request_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    output_dir: Mapped[str] = mapped_column(String(512), nullable=False)
    log_path: Mapped[str] = mapped_column(String(512), nullable=False)
    model_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    current_step: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_steps: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    
    def __repr__(self) -> str:
        return f"<FinetuneJob(id={self.id}, status={self.status})>"
