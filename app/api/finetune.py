"""
Fine-tuning job creation endpoints.
"""

import json
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.enums import JobStatus, FinetuneMode
from app.core.paths import ensure_dir
from app.db.crud import create_job
from app.db.session import get_db
from app.schemas.request import CreateFinetuneJobRequest
from app.schemas.response import CreateFinetuneJobResponse

router = APIRouter(prefix="/v1/finetune", tags=["finetune"])


def validate_request(request: CreateFinetuneJobRequest) -> dict:
    """
    Validate fine-tuning job request.
    
    Args:
        request: Request object
    
    Returns:
        Dictionary with validated parameters
    
    Raises:
        HTTPException: If validation fails
    """
    # Validate finetune_mode
    valid_modes = {FinetuneMode.lora.value, FinetuneMode.full.value}
    if request.finetune_mode not in valid_modes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"finetune_mode must be one of {valid_modes}, got {request.finetune_mode}",
        )
    
    # Validate train_data_path is not empty
    if not request.train_data_path or not request.train_data_path.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="train_data_path is required and cannot be empty",
        )
    
    # Validate prediction_length is positive
    if request.prediction_length <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="prediction_length must be positive",
        )
    
    # Validate other positive integers
    for field, value in [
        ("context_length", request.context_length),
        ("num_steps", request.num_steps),
        ("batch_size", request.batch_size),
        ("logging_steps", request.logging_steps),
    ]:
        if value <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{field} must be positive",
            )
    
    # Validate learning_rate is positive
    if request.learning_rate <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="learning_rate must be positive",
        )
    
    return {
        "model_id": request.model_id,
        "train_data_path": request.train_data_path,
        "val_data_path": request.val_data_path,
        "prediction_length": request.prediction_length,
        "context_length": request.context_length,
        "finetune_mode": request.finetune_mode,
        "learning_rate": request.learning_rate,
        "num_steps": request.num_steps,
        "batch_size": request.batch_size,
        "logging_steps": request.logging_steps,
        "output_root": request.output_root,
        "finetuned_ckpt_name": request.finetuned_ckpt_name,
        "device": request.device,
    }


@router.post(
    "/jobs",
    response_model=CreateFinetuneJobResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_finetune_job(
    request: CreateFinetuneJobRequest,
    db: Session = Depends(get_db),
) -> CreateFinetuneJobResponse:
    """
    Create a new fine-tuning job.
    
    The job is queued in the database but not started.
    
    Args:
        request: Fine-tuning job request
        db: Database session
    
    Returns:
        CreateFinetuneJobResponse with job_id and status
    
    Raises:
        HTTPException: If validation fails
    """
    # Validate request
    validated_params = validate_request(request)
    
    # Generate job_id
    job_id = str(uuid.uuid4())
    
    # Get settings
    settings = get_settings()
    
    # Determine output root
    output_root = Path(validated_params["output_root"]) if validated_params["output_root"] else settings.artifacts_root_resolved
    
    # Create job output directory
    job_output_dir = output_root / job_id
    ensure_dir(job_output_dir)
    
    # Write request.json
    request_json_path = job_output_dir / "request.json"
    with open(request_json_path, "w") as f:
        json.dump(validated_params, f, indent=2)
    
    # Define log path
    log_path = job_output_dir / "train.log"
    
    # Create job record in database
    db_job = create_job(
        db=db,
        job_id=job_id,
        status=JobStatus.queued.value,
        request_data=validated_params,
        output_dir=str(job_output_dir),
        log_path=str(log_path),
        max_steps=validated_params["num_steps"],
    )
    
    return CreateFinetuneJobResponse(
        job_id=job_id,
        status=JobStatus.queued.value,
    )
