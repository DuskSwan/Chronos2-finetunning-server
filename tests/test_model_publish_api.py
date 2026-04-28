"""Tests for /api/model/publish compatibility endpoint."""

import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import Settings
from app.db import crud
from app.db.models import Base
from app.db.session import get_db
from app.main import create_app


@pytest.fixture
def temp_base_dir():
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_settings(temp_base_dir):
    artifacts_dir = temp_base_dir / "artifacts"
    logs_dir = temp_base_dir / "logs"
    release_dir = temp_base_dir / "release"
    db_path = temp_base_dir / "test.db"

    artifacts_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    release_dir.mkdir(exist_ok=True)

    return Settings(
        host="127.0.0.1",
        port=8000,
        sqlite_db_path=str(db_path),
        artifacts_root=str(artifacts_dir),
        logs_root=str(logs_dir),
        release_path=str(release_dir),
        api_bearer_token="spec-token",
    )


@pytest.fixture
def test_db_session(test_settings):
    db_path = Path(test_settings.sqlite_db_path)
    db_url = f"sqlite:///{db_path.as_posix()}"
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    db = SessionLocal()
    yield db
    db.close()
    engine.dispose()


@pytest.fixture
def client(test_settings, test_db_session):
    def get_test_db():
        yield test_db_session

    with patch("app.main.get_settings", return_value=test_settings):
        with patch("app.api.model_publish.get_settings", return_value=test_settings):
            with patch("app.core.auth.get_settings", return_value=test_settings):
                with patch("app.main.initialize_worker", lambda *_args, **_kwargs: None):
                    app = create_app()
                    app.dependency_overrides[get_db] = get_test_db
                    with TestClient(app) as test_client:
                        yield test_client


def _auth_header() -> dict[str, str]:
    return {"Authorization": "Bearer spec-token"}


def test_publish_model_success(client: TestClient, test_db_session, temp_base_dir):
    job_id = "train_job_20260121103000"
    source_dir = temp_base_dir / "artifacts" / job_id
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "model.bin").write_text("mock model", encoding="utf-8")

    crud.create_job(
        db=test_db_session,
        job_id=job_id,
        status="queued",
        request_json="{}",
        output_dir=str(source_dir),
        log_path=str(temp_base_dir / "logs" / f"{job_id}.log"),
        max_steps=1,
    )
    crud.mark_job_completed(
        db=test_db_session,
        job_id=job_id,
        model_paths=[str(source_dir)],
        finished_at=datetime.now(timezone.utc),
    )

    response = client.post(
        "/api/model/publish",
        headers=_auth_header(),
        json={"user_id": 10001, "version": "1.0.0", "job_id": job_id},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 0
    assert body["message"] == "success"
    assert body["data"]["model_path"] == "models/user_10001/v1.0.0/train_job_20260121103000/model.bin"


def test_publish_model_invalid_version_format(client: TestClient):
    response = client.post(
        "/api/model/publish",
        headers=_auth_header(),
        json={"user_id": 10001, "version": "1.0", "job_id": "train_job_20260121103000"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 500
    assert body["message"] == "Value error, invalid version format, expected x.y.z"
    assert body["data"] is None


def test_publish_model_job_not_found(client: TestClient):
    response = client.post(
        "/api/model/publish",
        headers=_auth_header(),
        json={"user_id": 10001, "version": "1.0.0", "job_id": "not-exist"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 404
    assert body["message"] == "job_id not found"
    assert body["data"] is None
