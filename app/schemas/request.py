"""
Request schemas for fine-tuning API.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CreateFinetuneJobRequest(BaseModel):
    """Request schema for creating a fine-tuning job."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_id": "amazon/chronos-2",
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
                "device": "cpu",
            }
        }
    )
    
    model_id: str = Field(
        default="amazon/chronos-2",
        description="Model identifier",
    )
    train_data_path: str = Field(
        description="Path to training data file",
    )
    val_data_path: Optional[str] = Field(
        default=None,
        description="Path to validation data file",
    )
    prediction_length: int = Field(
        description="Prediction length for the model",
    )
    context_length: int = Field(
        default=512,
        description="Context length for the model",
    )
    finetune_mode: str = Field(
        default="lora",
        description="Fine-tuning mode: 'lora' or 'full'",
    )
    learning_rate: float = Field(
        default=1e-4,
        description="Learning rate for training",
    )
    num_steps: int = Field(
        default=1000,
        description="Number of training steps",
    )
    batch_size: int = Field(
        default=32,
        description="Training batch size",
    )
    logging_steps: int = Field(
        default=100,
        description="Logging frequency (steps)",
    )
    output_root: Optional[str] = Field(
        default=None,
        description="Output root directory (uses config default if null)",
    )
    finetuned_ckpt_name: str = Field(
        default="finetuned-ckpt",
        description="Name of the fine-tuned checkpoint",
    )
    device: str = Field(
        default="cpu",
        description="Device to use: 'cpu' or 'cuda'",
    )

    @field_validator("train_data_path")
    @classmethod
    def validate_train_data_path(cls, v: str) -> str:
        """Validate training data path is not empty."""
        if not v or not v.strip():
            raise ValueError("train_data_path cannot be empty")
        return v

    @field_validator("prediction_length")
    @classmethod
    def validate_prediction_length(cls, v: int) -> int:
        """Validate prediction length is positive."""
        if v <= 0:
            raise ValueError("prediction_length must be positive")
        return v

    @field_validator("context_length")
    @classmethod
    def validate_context_length(cls, v: int) -> int:
        """Validate context length is positive."""
        if v <= 0:
            raise ValueError("context_length must be positive")
        return v

    @field_validator("finetune_mode")
    @classmethod
    def validate_finetune_mode(cls, v: str) -> str:
        """Validate fine-tuning mode."""
        if v not in ("lora", "full"):
            raise ValueError("finetune_mode must be 'lora' or 'full'")
        return v

    @field_validator("learning_rate")
    @classmethod
    def validate_learning_rate(cls, v: float) -> float:
        """Validate learning rate is positive."""
        if v <= 0:
            raise ValueError("learning_rate must be positive")
        return v

    @field_validator("num_steps")
    @classmethod
    def validate_num_steps(cls, v: int) -> int:
        """Validate number of steps is positive."""
        if v <= 0:
            raise ValueError("num_steps must be positive")
        return v

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """Validate batch size is positive."""
        if v <= 0:
            raise ValueError("batch_size must be positive")
        return v

    @field_validator("logging_steps")
    @classmethod
    def validate_logging_steps(cls, v: int) -> int:
        """Validate logging steps is positive."""
        if v <= 0:
            raise ValueError("logging_steps must be positive")
        return v
