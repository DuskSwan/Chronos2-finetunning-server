"""
微调任务创建端点。
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
        ("logging_steps", request.logging_steps),
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
    创建新的微调任务。
    
    任务在数据库中排队但不启动。
    
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
    
    # 获取设置
    settings = get_settings()
    
    # 确定输出根目录
    output_root = Path(validated_params["output_root"]) if validated_params["output_root"] else settings.artifacts_root_resolved
    
    # 创建任务输出目录
    job_output_dir = output_root / job_id
    ensure_dir(job_output_dir)
    
    # 写入 request.json
    request_json_path = job_output_dir / "request.json"
    with open(request_json_path, "w") as f:
        json.dump(validated_params, f, indent=2)
    
    # 定义日志路径
    log_path = job_output_dir / "train.log"
    
    # 在数据库中创建任务记录
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
