"""
微调任务的状态枚举。
"""

from enum import Enum


class JobStatus(str, Enum):
    """任务状态枚举。"""
    
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class FinetuneMode(str, Enum):
    """微调模式枚举。"""
    
    lora = "lora"
    full = "full"
