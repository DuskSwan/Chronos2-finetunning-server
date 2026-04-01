"""
SQLAlchemy models for fine-tuning jobs.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import Column, DateTime, Float, Integer, String, Boolean, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class FinetuneJob(Base):
    """Fine-tune job model."""
    
    __tablename__ = "finetune_jobs"
    
    id: str = Column(String(36), primary_key=True)
    status: str = Column(String(20), nullable=False, default="queued")
    request_json: str = Column(Text, nullable=False)
    created_at: datetime = Column(DateTime, nullable=False, default=datetime.utcnow)
    started_at: Optional[datetime] = Column(DateTime, nullable=True)
    finished_at: Optional[datetime] = Column(DateTime, nullable=True)
    output_dir: str = Column(String(512), nullable=False)
    log_path: str = Column(String(512), nullable=False)
    model_path: Optional[str] = Column(String(512), nullable=True)
    error_message: Optional[str] = Column(Text, nullable=True)
    current_step: int = Column(Integer, nullable=False, default=0)
    max_steps: int = Column(Integer, nullable=False, default=0)
    last_loss: Optional[float] = Column(Float, nullable=True)
    cancel_requested: bool = Column(Boolean, nullable=False, default=False)
    
    def __repr__(self) -> str:
        return f"<FinetuneJob(id={self.id}, status={self.status})>"
