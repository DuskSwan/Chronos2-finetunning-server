"""Bearer 鉴权依赖（规范兼容路由专用）。"""

from fastapi import Header

from app.core.config import get_settings
from app.core.errors import ApiError


def require_bearer_token(authorization: str | None = Header(default=None)) -> None:
    """校验 Bearer Token。"""
    settings = get_settings()
    expected_token = settings.api_bearer_token
    if not expected_token:
        return

    if not authorization:
        raise ApiError(code=40101, message="unauthorized", http_status=401)

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or token != expected_token:
        raise ApiError(code=40101, message="unauthorized", http_status=401)

