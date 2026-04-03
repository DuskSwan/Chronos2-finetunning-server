"""
微调 API 的请求 Schema。
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
                "logging_steps": 100,
                "finetuned_ckpt_name": "finetuned-ckpt",
                "selected_columns": ["target"],
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
    logging_steps: int = Field(
        default=100,
        description="日志记录频率（步数）",
    )
    finetuned_ckpt_name: str = Field(
        default="finetuned-ckpt",
        description="微调检查点的名称",
    )
    selected_columns: Optional[list[str]] = Field(
        default=None,
        description="指定要使用的 CSV/Parquet 列名列表（为空则使用全部列）",
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

    @field_validator("logging_steps")
    @classmethod
    def validate_logging_steps(cls, v: int) -> int:
        """验证日志记录频率为正整数。"""
        if v <= 0:
            raise ValueError("日志记录频率必须是正整数")
        return v

    @field_validator("selected_columns")
    @classmethod
    def validate_selected_columns(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        """验证 selected_columns 非空且无重复。"""
        if v is None:
            return v
        if not v:
            raise ValueError("selected_columns 不能为空")
        unique = list(dict.fromkeys(v))
        if len(unique) != len(v):
            raise ValueError("selected_columns 不能包含重复项")
        return v
