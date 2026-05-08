"""规范兼容任务接口。"""

import json
import uuid

from fastapi import APIRouter, Depends
from fastapi.encoders import jsonable_encoder
from loguru import logger
from sqlalchemy.orm import Session

from app.core.auth import require_bearer_token
from app.core.config import get_settings
from app.core.errors import ApiError
from app.core.enums import JobStatus
from app.core.logging_utils import to_pretty_log
from app.core.paths import ensure_dir
from app.db.crud import create_job
from app.db.session import get_db
from app.schemas.request import CreateFinetuneJobRequest
from app.schemas.standard_response import ApiResponse, CreateTrainJobData, TrainJobStatusData
from app.services.job_service import get_job_detail
from app.services.queue_service import get_job_queue
from app.services.train_job_adapter import adapt_job_detail, normalize_api_error

router = APIRouter(prefix="/api/v1/train_jobs", tags=["train_jobs"])


def _build_request_payload(request: CreateFinetuneJobRequest) -> dict:
    payload = jsonable_encoder(request)
    settings = get_settings()
    payload["device"] = settings.device
    payload["logging_steps"] = settings.logging_steps
    payload["finetuned_ckpt_name"] = settings.finetuned_ckpt_name
    return payload


@router.post("", response_model=ApiResponse[CreateTrainJobData])
async def create_train_job(
    request: CreateFinetuneJobRequest,
    db: Session = Depends(get_db),
    _auth: None = Depends(require_bearer_token),
) -> ApiResponse[CreateTrainJobData]:
    """创建训练任务（规范兼容）。"""
    try:
        payload = _build_request_payload(request)
        job_id = str(uuid.uuid4())
        settings = get_settings()

        job_output_dir = settings.artifacts_root_resolved / job_id
        ensure_dir(job_output_dir)
        if settings.save_request_artifacts:
            request_json_path = job_output_dir / "request.json"
            with open(request_json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

        log_path = settings.logs_root_resolved / f"{job_id}.log"
        create_job(
            db=db,
            job_id=job_id,
            status=JobStatus.queued.value,
            request_json=json.dumps(payload, ensure_ascii=False),
            output_dir=str(job_output_dir),
            log_path=str(log_path),
            max_steps=int(payload.get("num_steps", 0)),
        )
        get_job_queue().enqueue(job_id)
        response = ApiResponse(
            code=0,
            message="success",
            data=CreateTrainJobData(job_id=job_id),
        )
        logger.info("create_train_job response:\n{}", to_pretty_log(response))
        return response
    except Exception as exc:
        raise normalize_api_error(exc) from exc


@router.get("/{job_id}", response_model=ApiResponse[TrainJobStatusData])
async def get_train_job_status(
    job_id: str,
    db: Session = Depends(get_db),
    _auth: None = Depends(require_bearer_token),
) -> ApiResponse[TrainJobStatusData]:
    """查询训练任务状态（规范兼容）。"""
    try:
        detail = get_job_detail(db, job_id)
        data = adapt_job_detail(detail)
        response = ApiResponse(code=0, message="success", data=data)
        logger.info("get_train_job_status response:\n{}", to_pretty_log(response))
        return response
    except Exception as exc:
        normalized = normalize_api_error(exc)
        if normalized.code == 40401:
            raise normalized
        raise ApiError(
            code=normalized.code,
            message=normalized.message,
            http_status=normalized.http_status,
        ) from exc
