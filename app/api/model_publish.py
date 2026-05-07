"""模型发布兼容接口。"""

import shutil
from pathlib import Path

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.auth import require_bearer_token
from app.core.config import get_settings
from app.core.enums import JobStatus
from app.core.paths import ensure_dir
from app.db.crud import get_job_by_id
from app.db.session import get_db
from app.schemas.request import ModelPublishRequest
from app.schemas.response import ModelPublishData, ModelPublishResponse

router = APIRouter(prefix="/api/model", tags=["model_publish"])


def _build_publish_subdir(user_id: int, version: str, job_id: str) -> Path:
    return Path(f"models/user_{user_id}/v{version}/{job_id}")


@router.post("/publish", response_model=ModelPublishResponse)
async def publish_model(
    request: ModelPublishRequest,
    db: Session = Depends(get_db),
    _auth: None = Depends(require_bearer_token),
) -> ModelPublishResponse:
    """发布模型，返回发布目录绝对路径。"""
    settings = get_settings()

    job = get_job_by_id(db, request.job_id)
    if not job:
        return ModelPublishResponse(code=404, message="job_id not found", data=None)

    if job.status != JobStatus.completed.value:
        return ModelPublishResponse(code=500, message="job is not completed", data=None)

    if not job.output_dir:
        return ModelPublishResponse(code=500, message="model source directory is missing", data=None)

    source_dir = Path(job.output_dir)
    if not source_dir.exists() or not source_dir.is_dir():
        return ModelPublishResponse(code=500, message="model source directory is missing", data=None)

    release_root = ensure_dir(settings.release_path_resolved)
    publish_subdir = _build_publish_subdir(
        user_id=request.user_id,
        version=request.version,
        job_id=request.job_id,
    )
    publish_dir = release_root / publish_subdir

    try:
        if publish_dir.exists():
            shutil.rmtree(publish_dir)
        publish_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_dir, publish_dir)
    except Exception:
        return ModelPublishResponse(code=500, message="internal error", data=None)

    return ModelPublishResponse(
        code=0,
        message="success",
        data=ModelPublishData(model_path=str(publish_dir.resolve())),
    )
