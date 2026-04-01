"""
微调 API 的响应 Schema。
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class HealthCheckResponse(BaseModel):
    """健康检查响应。"""
    
    status: str = Field(description="服务状态")


class CreateFinetuneJobResponse(BaseModel):
    """创建微调任务的响应。"""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "queued",
            }
        }
    )
    
    job_id: str = Field(description="唯一任务标识符")
    status: str = Field(description="任务状态")


class JobProgressResponse(BaseModel):
    """任务进度信息。"""
    current_step: int = Field(description="当前训练步数")
    max_steps: int = Field(description="最大训练步数")
    last_loss: Optional[float] = Field(default=None, description="最新损失值")


class JobDetailResponse(BaseModel):
    """任务详情响应。"""
    job_id: str = Field(description="唯一任务标识符")
    status: str = Field(description="任务状态")
    created_at: datetime = Field(description="任务创建时间")
    started_at: Optional[datetime] = Field(default=None, description="任务开始时间")
    finished_at: Optional[datetime] = Field(default=None, description="任务完成时间")
    progress: JobProgressResponse = Field(description="训练进度")
    error_message: Optional[str] = Field(default=None, description="错误信息")
    log_path: Optional[str] = Field(default=None, description="日志文件路径")
    model_path: Optional[str] = Field(default=None, description="模型输出路径")


class JobResultResponse(BaseModel):
    """任务结果响应。"""
    job_id: str = Field(description="唯一任务标识符")
    status: str = Field(description="任务状态")
    output_dir: str = Field(description="任务输出目录")
    model_path: Optional[str] = Field(default=None, description="模型输出路径")
    metrics: dict[str, Any] = Field(default_factory=dict, description="训练指标")


class JobListItemResponse(BaseModel):
    """任务列表条目。"""
    job_id: str = Field(description="唯一任务标识符")
    status: str = Field(description="任务状态")
    created_at: datetime = Field(description="任务创建时间")
    started_at: Optional[datetime] = Field(default=None, description="任务开始时间")
    finished_at: Optional[datetime] = Field(default=None, description="任务完成时间")


class JobListResponse(BaseModel):
    """任务列表响应。"""
    items: list[JobListItemResponse] = Field(description="任务列表")
