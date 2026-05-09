"""模型发布服务（含 metadata 生成）。"""

import json
import shutil
from pathlib import Path
from typing import Any

from app.db.models import FinetuneJob
from app.services.model_metadata_service import write_model_metadata


class ModelReleaseError(Exception):
    """模型发布异常。"""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def build_model_metadata(
    job: FinetuneJob,
    user_id: str,
    version: str,
) -> dict[str, Any]:
    """基于任务记录构建 metadata。"""
    try:
        request_payload = json.loads(job.request_json or "{}")
    except json.JSONDecodeError:
        request_payload = {}

    try:
        target_model_map = json.loads(job.target_model_map or "{}")
    except json.JSONDecodeError:
        target_model_map = {}

    selected_groups = request_payload.get("selected_groups") or []
    groups_with_model: list[dict[str, Any]] = []
    for group in selected_groups:
        target = str(group.get("target", "")).strip()
        covariates = group.get("covariates") or []
        model_dir = None
        abs_model_path = target_model_map.get(target)
        if abs_model_path:
            model_dir = Path(abs_model_path).name
        groups_with_model.append(
            {
                "target": target,
                "covariates": covariates,
                "model_dir": model_dir or f"finetuned-ckpt_{target}",
            }
        )

    return {
        "job_id": job.id,
        "user_id": user_id,
        "version": version,
        "model_type": "chronos-2",
        "prediction_length": request_payload.get("prediction_length"),
        "context_length": request_payload.get("context_length"),
        "selected_groups": groups_with_model,
        "created_from_train_request": {
            "finetune_mode": request_payload.get("finetune_mode"),
            "learning_rate": request_payload.get("learning_rate"),
            "num_steps": request_payload.get("num_steps"),
            "batch_size": request_payload.get("batch_size"),
        },
    }


def release_model_directory(
    source_dir: Path,
    release_dir: Path,
    job: FinetuneJob,
    user_id: str,
    version: str,
) -> Path:
    """复制模型目录并生成 metadata。"""
    if release_dir.exists():
        shutil.rmtree(release_dir)
    release_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, release_dir)

    metadata = build_model_metadata(job=job, user_id=user_id, version=version)
    write_model_metadata(release_dir, metadata)
    return release_dir

