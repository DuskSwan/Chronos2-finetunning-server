"""Tests for /api/model/infer/config and /api/model/infer/chunk endpoints."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import Settings
from app.db.models import Base
from app.db.session import get_db
from app.main import create_app


def _auth_header() -> dict[str, str]:
    return {"Authorization": "Bearer spec-token"}


class _FakeTensor:
    def __init__(self, values):
        self._values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        import numpy as np

        return np.asarray(self._values, dtype=float)


class _FakePipeline:
    def __init__(self, values):
        self._values = values

    def predict(self, inputs, prediction_length):
        _ = inputs
        _ = prediction_length
        return [_FakeTensor(self._values)]


def _build_client(temp_dir: Path) -> TestClient:
    artifacts_dir = temp_dir / "artifacts"
    logs_dir = temp_dir / "logs"
    release_dir = temp_dir / "release"
    db_path = temp_dir / "test.db"
    artifacts_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    release_dir.mkdir(exist_ok=True)

    test_settings = Settings(
        host="127.0.0.1",
        port=8000,
        sqlite_db_path=str(db_path),
        artifacts_root=str(artifacts_dir),
        logs_root=str(logs_dir),
        release_path=str(release_dir),
        api_bearer_token="spec-token",
        chunk_infer_cache_ttl_seconds=1800,
        chunk_infer_cache_max_tasks=2,
    )

    engine = create_engine(f"sqlite:///{db_path.as_posix()}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()

    def get_test_db():
        yield db

    with patch("app.main.get_settings", return_value=test_settings):
        with patch("app.core.auth.get_settings", return_value=test_settings):
            with patch("app.services.chunk_inference_service.get_settings", return_value=test_settings):
                with patch("app.main.initialize_worker", lambda *_args, **_kwargs: None):
                    app = create_app()
                    app.dependency_overrides[get_db] = get_test_db
                    client = TestClient(app)
                    client.__enter__()
                    client._test_db = db  # type: ignore[attr-defined]
                    client._test_engine = engine  # type: ignore[attr-defined]
                    return client


def _close_client(client: TestClient) -> None:
    db = getattr(client, "_test_db", None)
    engine = getattr(client, "_test_engine", None)
    client.__exit__(None, None, None)
    if db is not None:
        db.close()
    if engine is not None:
        engine.dispose()


def _build_model_root(temp_dir: Path, name: str) -> Path:
    model_root = temp_dir / "release" / "models" / "u1" / "v1.0.0" / name
    (model_root / "finetuned-ckpt_value1").mkdir(parents=True, exist_ok=True)
    (model_root / "metadata.json").write_text(
        (
            '{"selected_groups":[{"target":"value1","covariates":["value2"]}],'
            '"prediction_length":2,"context_length":2}'
        ),
        encoding="utf-8",
    )
    return model_root


def _segment_payload():
    return [
        {"time": 1, "value1": 1.0, "value2": 2.0},
        {"time": 2, "value1": 2.0, "value2": 3.0},
        {"time": 3, "value1": 3.0, "value2": 4.0},
        {"time": 4, "value1": 4.0, "value2": 5.0},
    ]


def test_infer_config_success():
    temp_dir = Path(tempfile.mkdtemp())
    client = _build_client(temp_dir)
    try:
        model_root = _build_model_root(temp_dir, "job_cfg")
        response = client.get(
            "/api/model/infer/config",
            headers=_auth_header(),
            params={"model_path": str(model_root.resolve())},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["code"] == 0
        assert body["data"]["prediction_length"] == 2
        assert body["data"]["context_length"] == 2
    finally:
        _close_client(client)
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_infer_chunk_reuse_and_release():
    temp_dir = Path(tempfile.mkdtemp())
    client = _build_client(temp_dir)
    try:
        model_root = _build_model_root(temp_dir, "job_chunk")
        with patch("app.services.chunk_inference_service.load_local_model", return_value=_FakePipeline([0.1, 0.2])) as mocked:
            first = client.post(
                "/api/model/infer/chunk",
                headers=_auth_header(),
                json={
                    "task_id": "task-1",
                    "model_path": str(model_root.resolve()),
                    "is_last_segment": False,
                    "segment": _segment_payload(),
                },
            )
            assert first.status_code == 200
            first_body = first.json()
            assert first_body["code"] == 0
            assert first_body["data"]["model_reused"] is False
            assert first_body["data"]["released"] is False
            assert mocked.call_count == 1

            second = client.post(
                "/api/model/infer/chunk",
                headers=_auth_header(),
                json={
                    "task_id": "task-1",
                    "model_path": str(model_root.resolve()),
                    "is_last_segment": True,
                    "segment": _segment_payload(),
                },
            )
            assert second.status_code == 200
            second_body = second.json()
            assert second_body["code"] == 0
            assert second_body["data"]["model_reused"] is True
            assert second_body["data"]["released"] is True
            assert mocked.call_count == 1

            third = client.post(
                "/api/model/infer/chunk",
                headers=_auth_header(),
                json={
                    "task_id": "task-1",
                    "model_path": str(model_root.resolve()),
                    "is_last_segment": False,
                    "segment": _segment_payload(),
                },
            )
            assert third.status_code == 200
            third_body = third.json()
            assert third_body["code"] == 0
            assert third_body["data"]["model_reused"] is False
            assert mocked.call_count == 2
    finally:
        _close_client(client)
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_infer_chunk_task_id_conflict_model_path():
    temp_dir = Path(tempfile.mkdtemp())
    client = _build_client(temp_dir)
    try:
        model_root_a = _build_model_root(temp_dir, "job_a")
        model_root_b = _build_model_root(temp_dir, "job_b")

        with patch("app.services.chunk_inference_service.load_local_model", return_value=_FakePipeline([0.1, 0.2])):
            ok_resp = client.post(
                "/api/model/infer/chunk",
                headers=_auth_header(),
                json={
                    "task_id": "task-conflict",
                    "model_path": str(model_root_a.resolve()),
                    "is_last_segment": False,
                    "segment": _segment_payload(),
                },
            )
            assert ok_resp.status_code == 200
            assert ok_resp.json()["code"] == 0

            bad_resp = client.post(
                "/api/model/infer/chunk",
                headers=_auth_header(),
                json={
                    "task_id": "task-conflict",
                    "model_path": str(model_root_b.resolve()),
                    "is_last_segment": False,
                    "segment": _segment_payload(),
                },
            )
            assert bad_resp.status_code == 200
            body = bad_resp.json()
            assert body["code"] == 409
            assert body["message"] == "task_id conflicts with another model_path"
    finally:
        _close_client(client)
        shutil.rmtree(temp_dir, ignore_errors=True)
