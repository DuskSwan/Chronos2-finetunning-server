"""
CRUD operations for fine-tuning jobs.
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
    Create a new fine-tuning job record.
    
    Args:
        db: Database session
        job_id: Unique job identifier
        status: Job status (e.g., "queued")
        request_data: Request parameters as dict
        output_dir: Output directory path
        log_path: Log file path
        max_steps: Maximum training steps
    
    Returns:
        Created FinetuneJob instance
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
    Get a job by ID.
    
    Args:
        db: Database session
        job_id: Job identifier
    
    Returns:
        FinetuneJob if found, None otherwise
    """
    return db.query(FinetuneJob).filter(FinetuneJob.id == job_id).first()


def list_jobs(
    db: Session,
    skip: int = 0,
    limit: int = 100,
) -> list[FinetuneJob]:
    """
    List all jobs with pagination.
    
    Args:
        db: Database session
        skip: Number of jobs to skip
        limit: Maximum number of jobs to return
    
    Returns:
        List of FinetuneJob instances
    """
    return db.query(FinetuneJob).offset(skip).limit(limit).all()
