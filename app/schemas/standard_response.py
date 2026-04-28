"""规范兼容接口的响应模型。"""

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    """统一响应包装。"""

    code: int = Field(description="业务状态码，0 为成功")
    message: str = Field(description="结果描述信息")
    data: T = Field(description="业务数据主体")


class CreateTrainJobData(BaseModel):
    """创建任务响应数据。"""

    job_id: str = Field(description="任务唯一标识")


class LossData(BaseModel):
    """任务损失信息。"""

    steps: list[int] = Field(default_factory=list, description="训练步数序列")
    values: list[float] = Field(default_factory=list, description="每步对应 loss 值")
    current_loss: float = Field(default=0.0, description="最新一次 loss 值")


class TrainJobStatusData(BaseModel):
    """任务状态查询响应数据。"""

    job_id: str = Field(description="任务唯一标识")
    is_completed: bool = Field(description="任务是否完成")
    status: str = Field(description="任务状态")
    loss_data: LossData = Field(description="loss 统一封装")
    duration: int = Field(description="任务耗时（秒）")

