"""模型推理接口。"""

from fastapi import APIRouter, Depends

from app.core.auth import require_bearer_token
from app.schemas.request import ModelInferRequest
from app.schemas.response import ModelInferResponse

router = APIRouter(prefix="/api/model", tags=["inference"])


@router.post("/infer", response_model=ModelInferResponse)
async def infer_model(
    request: ModelInferRequest,
    _auth: None = Depends(require_bearer_token),
) -> ModelInferResponse:
    """模型推理接口（第一部分：先完成契约与鉴权接入）。"""
    _ = request
    return ModelInferResponse(
        code=501,
        message="inference service not implemented yet",
        data=None,
    )
