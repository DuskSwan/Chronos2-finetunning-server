"""模型推理接口。"""

from fastapi import APIRouter, Depends

from app.core.auth import require_bearer_token
from app.schemas.request import ModelInferRequest
from app.schemas.response import ModelInferData, ModelInferResponse
from app.services.inference_service import InferenceError, run_inference

router = APIRouter(prefix="/api/model", tags=["inference"])


@router.post("/infer", response_model=ModelInferResponse)
async def infer_model(
    request: ModelInferRequest,
    _auth: None = Depends(require_bearer_token),
) -> ModelInferResponse:
    """模型推理接口。"""
    try:
        predictions = run_inference(
            model_path=request.model_path,
            cov_group=request.cov_group,
            prediction_length=request.prediction_length,
            csv_path=request.csv_path,
        )
    except InferenceError as exc:
        return ModelInferResponse(code=exc.code, message=exc.message, data=None)

    return ModelInferResponse(
        code=0,
        message="success",
        data=ModelInferData(predictions=predictions),
    )
