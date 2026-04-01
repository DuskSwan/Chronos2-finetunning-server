"""
Response schemas for fine-tuning API.
"""

from pydantic import BaseModel, ConfigDict, Field


class HealthCheckResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(description="Service status")


class CreateFinetuneJobResponse(BaseModel):
    """Response for creating a fine-tuning job."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "queued",
            }
        }
    )
    
    job_id: str = Field(description="Unique job identifier")
    status: str = Field(description="Job status")
