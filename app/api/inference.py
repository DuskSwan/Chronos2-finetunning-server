"""模型推理接口。"""

from fastapi import APIRouter, Depends
from loguru import logger

from app.core.auth import require_bearer_token
from app.core.logging_utils import to_pretty_log
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
            context_length=request.context_length,
            csv_path=request.csv_path,
        )
    except InferenceError as exc:
        response = ModelInferResponse(code=exc.code, message=exc.message, data=None)
        logger.info("infer_model response:\n{}", to_pretty_log(response))
        return response

    response = ModelInferResponse(
        code=0,
        message="success",
        data=ModelInferData(predictions=predictions),
    )
    logger.info("infer_model response:\n{}", to_pretty_log(response))
    return response
