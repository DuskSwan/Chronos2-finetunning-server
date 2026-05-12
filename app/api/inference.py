"""模型推理接口。"""

from fastapi import APIRouter, Depends
from loguru import logger

from app.core.auth import require_bearer_token
from app.core.logging_utils import to_pretty_log
from app.schemas.request import ModelInferChunkRequest, ModelInferConfigRequest, ModelInferRequest
from app.schemas.response import (
    ModelInferChunkData,
    ModelInferChunkResponse,
    ModelInferConfigData,
    ModelInferConfigResponse,
    ModelInferData,
    ModelInferResponse,
)
from app.services.chunk_inference_service import get_model_infer_config, run_chunk_inference
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


@router.get("/infer/config", response_model=ModelInferConfigResponse)
async def infer_model_config(
    request: ModelInferConfigRequest = Depends(),
    _auth: None = Depends(require_bearer_token),
) -> ModelInferConfigResponse:
    """查询模型推理默认配置。"""
    try:
        prediction_length, context_length = get_model_infer_config(request.model_path)
    except InferenceError as exc:
        response = ModelInferConfigResponse(code=exc.code, message=exc.message, data=None)
        logger.info("infer_model_config response:\n{}", to_pretty_log(response))
        return response

    response = ModelInferConfigResponse(
        code=0,
        message="success",
        data=ModelInferConfigData(
            model_path=request.model_path,
            prediction_length=prediction_length,
            context_length=context_length,
        ),
    )
    logger.info("infer_model_config response:\n{}", to_pretty_log(response))
    return response


@router.post("/infer/chunk", response_model=ModelInferChunkResponse)
async def infer_model_chunk(
    request: ModelInferChunkRequest,
    _auth: None = Depends(require_bearer_token),
) -> ModelInferChunkResponse:
    """模型分段推理接口。"""
    try:
        predictions, model_reused = run_chunk_inference(
            task_id=request.task_id,
            model_path=request.model_path,
            segment=request.segment,
            is_last_segment=request.is_last_segment,
        )
    except InferenceError as exc:
        response = ModelInferChunkResponse(code=exc.code, message=exc.message, data=None)
        logger.info("infer_model_chunk response:\n{}", to_pretty_log(response))
        return response

    response = ModelInferChunkResponse(
        code=0,
        message="success",
        data=ModelInferChunkData(
            task_id=request.task_id,
            predictions=predictions,
            model_reused=model_reused,
            released=request.is_last_segment,
        ),
    )
    logger.info("infer_model_chunk response:\n{}", to_pretty_log(response))
    return response
