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
    model_paths: Optional[list[str]] = Field(default=None, description="模型输出路径列表")
    target_model_map: Optional[dict[str, str]] = Field(default=None, description="target 到模型路径映射")
    output_dir: Optional[str] = Field(default=None, description="任务输出目录")
    metrics: dict[str, Any] = Field(default_factory=dict, description="loss 曲线数据")


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


class CancelJobResponse(BaseModel):
    """取消任务响应。"""
    job_id: str = Field(description="唯一任务标识符")
    status: str = Field(description="任务状态")
    cancel_requested: bool = Field(description="是否已请求取消")
    message: Optional[str] = Field(default=None, description="取消结果说明")


class DeleteJobResponse(BaseModel):
    """删除单任务响应。"""
    job_id: str = Field(description="唯一任务标识符")
    deleted: bool = Field(description="是否删除成功")
    removed_from_queue: bool = Field(description="是否从队列移除")
    files_deleted: int = Field(description="删除的本地文件/目录数量")
    message: Optional[str] = Field(default=None, description="删除结果说明")


class BatchDeleteJobsResponse(BaseModel):
    """批量删除任务响应。"""
    matched_jobs: int = Field(description="匹配到的任务数量")
    deleted_jobs: int = Field(description="成功删除的任务数量")
    skipped_running_jobs: int = Field(description="跳过的 running 任务数量")
    removed_from_queue: int = Field(description="从队列移除的任务数量")
    files_deleted: int = Field(description="删除的本地文件/目录数量")
    message: Optional[str] = Field(default=None, description="删除结果说明")


class ReleaseModelData(BaseModel):
    """发布模型的数据体。"""

    model_path: str = Field(description="发布后的模型目录绝对路径")


class ReleaseModelResponse(BaseModel):
    """发布模型统一响应。"""

    code: int = Field(description="业务状态码，0 表示成功")
    message: str = Field(description="结果描述")
    data: ReleaseModelData = Field(description="响应数据体")


class ModelPublishData(BaseModel):
    """模型发布接口 data。"""

    model_path: str = Field(description="发布后的模型目录绝对路径")


class ModelPublishResponse(BaseModel):
    """模型发布接口响应。"""

    code: int = Field(description="业务状态码")
    message: str = Field(description="结果描述")
    data: ModelPublishData | None = Field(description="响应数据")


class InferPredictionItem(BaseModel):
    """单个 target 的预测结果。"""

    target: str = Field(description="目标列名")
    prediction: list[float] = Field(description="预测值列表")
    actual: list[float] = Field(description="真实值列表，长度与 prediction 相同")


class ModelInferData(BaseModel):
    """模型推理接口 data。"""

    predictions: list[InferPredictionItem] = Field(description="按 cov_group 顺序返回的预测结果")


class ModelInferResponse(BaseModel):
    """模型推理接口响应。"""

    code: int = Field(description="业务状态码")
    message: str = Field(description="结果描述")
    data: ModelInferData | None = Field(description="响应数据")


class ModelInfoData(BaseModel):
    """模型信息接口 data。"""

    model_path: str = Field(description="模型目录绝对路径")
    targets: list[str] = Field(description="可预测目标列表")
    selected_groups: list[dict[str, Any]] = Field(description="训练分组")
    prediction_length: int = Field(description="默认预测长度")
    context_length: int = Field(description="默认上下文长度")


class ModelInfoResponse(BaseModel):
    """模型信息接口响应。"""

    code: int = Field(description="业务状态码")
    message: str = Field(description="结果描述")
    data: ModelInfoData | None = Field(description="响应数据")
