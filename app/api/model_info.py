"""模型信息接口。"""

from pathlib import Path

from fastapi import APIRouter, Depends
from loguru import logger

from app.core.auth import require_bearer_token
from app.core.logging_utils import to_pretty_log
from app.schemas.response import ModelInfoData, ModelInfoResponse
from app.services.model_metadata_service import ModelMetadataError, load_model_metadata

router = APIRouter(prefix="/api/model", tags=["model_info"])


@router.get("/info", response_model=ModelInfoResponse)
async def get_model_info(
    model_path: str,
    _auth: None = Depends(require_bearer_token),
) -> ModelInfoResponse:
    release_model_dir = Path(model_path)
    if not release_model_dir.exists() or not release_model_dir.is_dir():
        response = ModelInfoResponse(code=404, message="model path not found", data=None)
        logger.info("get_model_info response:\n{}", to_pretty_log(response))
        return response

    try:
        metadata = load_model_metadata(release_model_dir)
    except ModelMetadataError as exc:
        response = ModelInfoResponse(code=exc.code, message=exc.message, data=None)
        logger.info("get_model_info response:\n{}", to_pretty_log(response))
        return response

    selected_groups = metadata.get("selected_groups") or []
    targets = [str(item.get("target", "")).strip() for item in selected_groups if item.get("target")]

    response = ModelInfoResponse(
        code=0,
        message="success",
        data=ModelInfoData(
            model_path=str(release_model_dir.resolve()),
            targets=targets,
            selected_groups=selected_groups,
            prediction_length=int(metadata.get("prediction_length")),
            context_length=int(metadata.get("context_length")),
        ),
    )
    logger.info("get_model_info response:\n{}", to_pretty_log(response))
    return response

