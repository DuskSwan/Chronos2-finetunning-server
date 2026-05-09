"""模型 metadata 读写服务。"""

import json
from pathlib import Path
from typing import Any


class ModelMetadataError(Exception):
    """metadata 读写异常。"""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def write_model_metadata(model_path: str | Path, metadata: dict[str, Any]) -> Path:
    """将 metadata 写入模型目录。"""
    model_dir = Path(model_path)
    model_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = model_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata_path


def load_model_metadata(model_path: str | Path) -> dict[str, Any]:
    """读取并校验模型目录下的 metadata.json。"""
    model_dir = Path(model_path)
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        raise ModelMetadataError(404, "metadata.json not found")

    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelMetadataError(400, f"metadata.json is invalid json: {exc.msg}") from exc

    for key in ("selected_groups", "prediction_length", "context_length"):
        if key not in payload:
            raise ModelMetadataError(400, f"metadata.json missing required field: {key}")
    return payload

