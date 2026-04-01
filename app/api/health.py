"""
Health check endpoint.
"""

from fastapi import APIRouter

from app.schemas.response import HealthCheckResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """
    Health check endpoint.
    
    Returns:
        HealthCheckResponse with status "ok"
    """
    return HealthCheckResponse(status="ok")
