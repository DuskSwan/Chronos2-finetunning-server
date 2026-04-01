"""
Status enumerations for fine-tuning jobs.
"""

from enum import Enum


class JobStatus(str, Enum):
    """Job status enumeration."""
    
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class FinetuneMode(str, Enum):
    """Fine-tune mode enumeration."""
    
    lora = "lora"
    full = "full"
