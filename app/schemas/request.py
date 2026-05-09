"""
微调 API 的请求 Schema。
"""

from typing import Optional
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SelectedGroup(BaseModel):
    """目标列与协变量列的分组定义。"""

    target: str = Field(
        description="目标列名",
    )
    covariates: list[str] = Field(
        default_factory=list,
        description="协变量列名列表",
    )

    @field_validator("target")
    @classmethod
    def validate_target(cls, v: str) -> str:
        """验证 target 非空。"""
        if not v or not v.strip():
            raise ValueError("selected_groups.target 不能为空")
        return v

    @field_validator("covariates")
    @classmethod
    def validate_covariates(cls, v: list[str]) -> list[str]:
        """验证 covariates 非空字符串且无重复。"""
        if any((not item) or (not item.strip()) for item in v):
            raise ValueError("selected_groups.covariates 不能包含空值")
        unique = list(dict.fromkeys(v))
        if len(unique) != len(v):
            raise ValueError("selected_groups.covariates 不能包含重复项")
        return v

    @model_validator(mode="after")
    def validate_group(self) -> "SelectedGroup":
        """验证 target 不与 covariates 冲突。"""
        if self.target in self.covariates:
            raise ValueError("selected_groups.target 不能出现在 covariates 中")
        return self


class CreateFinetuneJobRequest(BaseModel):
    """创建微调任务的请求 Schema。"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "train_data_path": "/path/to/train.csv",
                "val_data_path": "/path/to/val.csv",
                "prediction_length": 96,
                "context_length": 512,
                "finetune_mode": "lora",
                "learning_rate": 0.0001,
                "num_steps": 1000,
                "batch_size": 32,
                "selected_groups": [
                    {
                        "target": "target",
                        "covariates": ["feature1", "feature2"],
                    }
                ],
            }
        }
    )
    
    train_data_path: str = Field(
        description="训练数据文件路径",
    )
    val_data_path: Optional[str] = Field(
        default=None,
        description="验证数据文件路径",
    )
    prediction_length: int = Field(
        description="模型的预测长度",
    )
    context_length: int = Field(
        default=512,
        description="模型的上下文长度",
    )
    finetune_mode: str = Field(
        default="lora",
        description="微调模式: 'lora' 或 'full'",
    )
    learning_rate: float = Field(
        default=1e-4,
        description="训练学习率",
    )
    num_steps: int = Field(
        default=1000,
        description="训练步数",
    )
    batch_size: int = Field(
        default=32,
        description="训练批大小",
    )
    selected_groups: Optional[list[SelectedGroup]] = Field(
        default=None,
        description=(
            "目标列与协变量列的分组列表（为空则使用全部列）。"
            "每个分组包含 target 与 covariates。"
        ),
    )

    @field_validator("train_data_path")
    @classmethod
    def validate_train_data_path(cls, v: str) -> str:
        """验证训练数据路径非空。"""
        if not v or not v.strip():
            raise ValueError("训练数据路径不能为空")
        return v

    @field_validator("prediction_length")
    @classmethod
    def validate_prediction_length(cls, v: int) -> int:
        """验证预测长度为正整数。"""
        if v <= 0:
            raise ValueError("预测长度必须是正整数")
        return v

    @field_validator("context_length")
    @classmethod
    def validate_context_length(cls, v: int) -> int:
        """验证上下文长度为正整数。"""
        if v <= 0:
            raise ValueError("上下文长度必须是正整数")
        return v

    @field_validator("finetune_mode")
    @classmethod
    def validate_finetune_mode(cls, v: str) -> str:
        """验证微调模式。"""
        if v not in ("lora", "full"):
            raise ValueError("微调模式必须是 'lora' 或 'full'")
        return v

    @field_validator("learning_rate")
    @classmethod
    def validate_learning_rate(cls, v: float) -> float:
        """验证学习率为正数。"""
        if v <= 0:
            raise ValueError("学习率必须是正数")
        return v

    @field_validator("num_steps")
    @classmethod
    def validate_num_steps(cls, v: int) -> int:
        """验证训练步数为正整数。"""
        if v <= 0:
            raise ValueError("训练步数必须是正整数")
        return v

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """验证批大小为正整数。"""
        if v <= 0:
            raise ValueError("批大小必须是正整数")
        return v

    @field_validator("selected_groups")
    @classmethod
    def validate_selected_groups(
        cls,
        v: Optional[list[SelectedGroup]],
    ) -> Optional[list[SelectedGroup]]:
        """验证 selected_groups 非空且目标列不重复。"""
        if v is None:
            return v
        if not v:
            raise ValueError("selected_groups 不能为空")
        targets = [group.target for group in v]
        unique_targets = list(dict.fromkeys(targets))
        if len(unique_targets) != len(targets):
            raise ValueError("selected_groups.target 不能包含重复项")
        return v


class ReleaseModelRequest(BaseModel):
    """发布模型请求。"""

    user_id: Optional[str] = Field(default=None, description="用户 ID")
    job_id: Optional[str] = Field(default=None, description="任务 ID")
    version: Optional[str] = Field(default=None, description="版本号")


class ModelPublishRequest(BaseModel):
    """模型发布兼容接口请求。"""

    user_id: int = Field(description="用户唯一标识 ID")
    version: str = Field(description="语义化版本号，格式 x.y.z")
    job_id: str = Field(description="训练任务 ID")

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        if not re.fullmatch(r"\d+\.\d+\.\d+", v):
            raise ValueError("invalid version format, expected x.y.z")
        return v

    @field_validator("job_id")
    @classmethod
    def validate_job_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("job_id is required")
        return v.strip()


class InferCovGroup(BaseModel):
    """推理 cov_group 分组。"""

    target: str = Field(description="目标列名")
    covariates: list[str] = Field(default_factory=list, description="协变量列名列表")

    @field_validator("target")
    @classmethod
    def validate_target(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("cov_group.target is required")
        return v.strip()

    @field_validator("covariates")
    @classmethod
    def validate_covariates(cls, v: list[str]) -> list[str]:
        if any((not item) or (not item.strip()) for item in v):
            raise ValueError("cov_group.covariates cannot contain empty values")
        unique = list(dict.fromkeys(v))
        if len(unique) != len(v):
            raise ValueError("cov_group.covariates cannot contain duplicates")
        return [item.strip() for item in v]

    @model_validator(mode="after")
    def validate_group(self) -> "InferCovGroup":
        if self.target in self.covariates:
            raise ValueError("cov_group.target cannot appear in covariates")
        return self


class ModelInferRequest(BaseModel):
    """模型推理接口请求。"""

    model_path: str = Field(description="发布后的模型绝对路径")
    cov_group: Optional[list[InferCovGroup]] = Field(
        default=None,
        description="预测分组列表，每组包含 target 与 covariates；可选，默认取 metadata",
    )
    prediction_length: Optional[int] = Field(default=None, description="预测步数；可选，默认取 metadata")
    context_length: Optional[int] = Field(default=None, description="上下文长度；可选，默认取 metadata")
    csv_path: str = Field(description="推理数据 CSV 文件路径")

    @field_validator("model_path", "csv_path")
    @classmethod
    def validate_required_path(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("path is required")
        return v.strip()

    @field_validator("prediction_length")
    @classmethod
    def validate_prediction_length(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return v
        if v <= 0:
            raise ValueError("prediction_length must be a positive integer")
        return v

    @field_validator("context_length")
    @classmethod
    def validate_context_length(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return v
        if v <= 0:
            raise ValueError("context_length must be a positive integer")
        return v

    @field_validator("cov_group")
    @classmethod
    def validate_cov_group(cls, v: Optional[list[InferCovGroup]]) -> Optional[list[InferCovGroup]]:
        if v is None:
            return v
        if not v:
            raise ValueError("cov_group cannot be empty")
        targets = [item.target for item in v]
        if len(set(targets)) != len(targets):
            raise ValueError("cov_group.target cannot contain duplicates")
        return v
