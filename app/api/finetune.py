"""
微调任务创建端点。
"""

import json
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session
from loguru import logger

from app.core.config import get_settings
from app.core.enums import JobStatus, FinetuneMode
from app.core.paths import ensure_dir
from app.db.crud import create_job, get_job_by_id
from app.db.session import get_db
from app.schemas.request import CreateFinetuneJobRequest, ReleaseModelRequest
from app.schemas.response import (
    CreateFinetuneJobResponse,
    JobDetailResponse,
    JobListResponse,
    CancelJobResponse,
    ReleaseModelResponse,
    ReleaseModelData,
)
from app.services.job_service import (
    get_job_detail,
    read_job_log,
    list_job_summaries,
    request_cancel_job,
)
from app.services.queue_service import get_job_queue

router = APIRouter(prefix="/v1/finetune", tags=["finetune"])


def validate_request(request: CreateFinetuneJobRequest) -> dict:
    """
    验证微调任务请求。
    
    参数：
        request: 请求对象
    
    返回：
        包含验证参数的字典
    
    抛出：
        HTTPException: 如果验证失败
    """
    # 验证 finetune_mode
    valid_modes = {FinetuneMode.lora.value, FinetuneMode.full.value}
    if request.finetune_mode not in valid_modes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"finetune_mode 必须是 {valid_modes} 之一，得到 {request.finetune_mode}",
        )
    
    # 验证 train_data_path 非空
    if not request.train_data_path or not request.train_data_path.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="train_data_path 是必需的且不能为空",
        )
    
    # 验证 prediction_length 为正数
    if request.prediction_length <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="prediction_length 必须为正数",
        )
    
    # 验证其他正整数字段
    for field, value in [
        ("context_length", request.context_length),
        ("num_steps", request.num_steps),
        ("batch_size", request.batch_size),
    ]:
        if value <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{field} 必须为正数",
            )
    
    # 验证 learning_rate 为正数
    if request.learning_rate <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="learning_rate 必须为正数",
        )
    
    return {
        "train_data_path": request.train_data_path,
        "val_data_path": request.val_data_path,
        "prediction_length": request.prediction_length,
        "context_length": request.context_length,
        "finetune_mode": request.finetune_mode,
        "learning_rate": request.learning_rate,
        "num_steps": request.num_steps,
        "batch_size": request.batch_size,
        "selected_groups": request.selected_groups,
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
    创建新的微调任务。
    
    任务在数据库中排队后，会自动被后台 worker 消费。
    
    参数：
        request: 微调任务请求
        db: 数据库会话
    
    返回：
        包含 job_id 和 status 的 CreateFinetuneJobResponse
    
    抛出：
        HTTPException: 如果验证失败
    """
    # 验证请求
    validated_params = validate_request(request)
    
    # 生成 job_id
    job_id = str(uuid.uuid4())
    logger.info(f"收到新的训练任务请求，已创建任务 ID: {job_id}")
    
    # 获取设置
    settings = get_settings()

    # 设备统一从配置读取，不再从请求传入
    validated_params["device"] = settings.device
    validated_params["logging_steps"] = settings.logging_steps
    validated_params["finetuned_ckpt_name"] = settings.finetuned_ckpt_name
    
    # 创建任务输出目录
    job_output_dir = settings.artifacts_root_resolved / job_id
    ensure_dir(job_output_dir)
    
    # 统一转成可 JSON 序列化结构
    validated_params = jsonable_encoder(validated_params)

    # 写入 request.json（可配置关闭）
    if settings.save_request_artifacts:
        request_json_path = job_output_dir / "request.json"
        with open(request_json_path, "w") as f:
            json.dump(validated_params, f, indent=2)
    
    # 定义日志路径（统一在 logs_root 下）
    log_path = settings.logs_root_resolved / f"{job_id}.log"
    
    # 在数据库中创建任务记录
    db_job = create_job(
        db=db,
        job_id=job_id,
        status=JobStatus.queued.value,
        request_json=json.dumps(validated_params),
        output_dir=str(job_output_dir),
        log_path=str(log_path),
        max_steps=validated_params["num_steps"],
    )
    
    # 将任务入队，worker 会自动处理
    queue = get_job_queue()
    queue.enqueue(job_id)
    
    return CreateFinetuneJobResponse(
        job_id=job_id,
        status=JobStatus.queued.value,
    )


@router.get(
    "/jobs/{job_id}",
    response_model=JobDetailResponse,
)
async def get_finetune_job_detail(
    job_id: str,
    db: Session = Depends(get_db),
) -> JobDetailResponse:
    """查询任务详情。"""
    return get_job_detail(db, job_id)


@router.get(
    "/jobs/{job_id}/logs",
    response_class=PlainTextResponse,
)
async def get_finetune_job_logs(
    job_id: str,
    tail: int | None = None,
    db: Session = Depends(get_db),
) -> PlainTextResponse:
    """查询任务日志。"""
    log_text = read_job_log(db, job_id, tail=tail)
    return PlainTextResponse(log_text)


@router.get(
    "/jobs",
    response_model=JobListResponse,
)
async def list_finetune_jobs(
    limit: int = 20,
    db: Session = Depends(get_db),
) -> JobListResponse:
    """查询任务列表（最近若干条）。"""
    return list_job_summaries(db, limit=limit)


@router.post(
    "/jobs/{job_id}/cancel",
    response_model=CancelJobResponse,
)
async def cancel_finetune_job(
    job_id: str,
    db: Session = Depends(get_db),
) -> CancelJobResponse:
    """取消任务（协作式取消）。"""
    return request_cancel_job(db, job_id)


@router.post(
    "/jobs/release",
    response_model=ReleaseModelResponse,
)
async def release_finetuned_model(
    request: ReleaseModelRequest,
    db: Session = Depends(get_db),
) -> ReleaseModelResponse:
    """发布已训练完成的模型目录。"""
    if (
        not request.user_id
        or not request.user_id.strip()
        or not request.job_id
        or not request.job_id.strip()
        or not request.version
        or not request.version.strip()
    ):
        return ReleaseModelResponse(
            code=401,
            message="invalid parameter",
            data=ReleaseModelData(model_path=""),
        )

    user_id = request.user_id.strip()
    job_id = request.job_id.strip()
    version = request.version.strip()

    job = get_job_by_id(db, job_id)
    if not job:
        return ReleaseModelResponse(
            code=404,
            message=f"job not found: {job_id}",
            data=ReleaseModelData(model_path=""),
        )

    if job.status != JobStatus.completed.value:
        return ReleaseModelResponse(
            code=409,
            message=f"job status is not completed: {job.status}",
            data=ReleaseModelData(model_path=""),
        )

    if not job.output_dir:
        return ReleaseModelResponse(
            code=404,
            message="model directory not found",
            data=ReleaseModelData(model_path=""),
        )

    source_dir = Path(job.output_dir)
    if not source_dir.exists() or not source_dir.is_dir():
        return ReleaseModelResponse(
            code=404,
            message=f"model directory not found: {source_dir}",
            data=ReleaseModelData(model_path=""),
        )

    release_name = f"{user_id}_{job_id}_{version}"
    settings = get_settings()
    release_root = ensure_dir(settings.release_path_resolved)
    release_dir = release_root / release_name

    try:
        if release_dir.exists():
            shutil.rmtree(release_dir)
        shutil.copytree(src=source_dir, dst=release_dir)
    except Exception as exc:
        return ReleaseModelResponse(
            code=500,
            message=f"release failed: {exc}",
            data=ReleaseModelData(model_path=""),
        )

    return ReleaseModelResponse(
        code=0,
        message="success",
        data=ReleaseModelData(model_path=str(release_dir.resolve())),
    )
