"""
健康检查端点。
"""

from fastapi import APIRouter
from loguru import logger

from app.core.logging_utils import to_pretty_log
from app.schemas.response import HealthCheckResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """
    健康检查端点。
    
    返回：
        包含状态 "ok" 的 HealthCheckResponse
    """
    response = HealthCheckResponse(status="ok")
    logger.info("health_check response:\n{}", to_pretty_log(response))
    return response
