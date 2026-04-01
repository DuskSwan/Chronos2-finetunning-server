"""
微调 API 的响应 Schema。
"""

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
