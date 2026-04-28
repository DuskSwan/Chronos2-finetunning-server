"""Tests for spec-compatible train job APIs."""

import shutil
import tempfile
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
    db_path = temp_base_dir / "test.db"

    artifacts_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    return Settings(
        host="127.0.0.1",
        port=8000,
        sqlite_db_path=str(db_path),
        artifacts_root=str(artifacts_dir),
        logs_root=str(logs_dir),
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
        with patch("app.api.train_jobs.get_settings", return_value=test_settings):
            with patch("app.core.auth.get_settings", return_value=test_settings):
                with patch("app.main.initialize_worker", lambda *_args, **_kwargs: None):
                    app = create_app()
                    app.dependency_overrides[get_db] = get_test_db
                    with TestClient(app) as test_client:
                        yield test_client


def _auth_header() -> dict[str, str]:
    return {"Authorization": "Bearer spec-token"}


def test_create_train_job_success(client: TestClient):
    payload = {
        "train_data_path": "/path/to/train.csv",
        "prediction_length": 96,
    }
    response = client.post("/api/v1/train_jobs", json=payload, headers=_auth_header())
    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 0
    assert body["message"] == "success"
    assert "job_id" in body["data"]


def test_create_train_job_unauthorized(client: TestClient):
    payload = {
        "train_data_path": "/path/to/train.csv",
        "prediction_length": 96,
    }
    response = client.post("/api/v1/train_jobs", json=payload)
    assert response.status_code == 401
    body = response.json()
    assert body["code"] == 40101
    assert body["message"] == "unauthorized"


def test_create_train_job_invalid_parameter(client: TestClient):
    payload = {"prediction_length": 96}
    response = client.post("/api/v1/train_jobs", json=payload, headers=_auth_header())
    assert response.status_code == 400
    body = response.json()
    assert body["code"] == 40001
    assert body["message"] == "invalid parameter"


def test_get_train_job_status_success_and_status_mapping(client: TestClient, test_db_session):
    job_id = "job-spec-status-1"
    crud.create_job(
        db=test_db_session,
        job_id=job_id,
        status="queued",
        request_json='{"selected_groups":[{"target":"target","covariates":[]}]}',
        output_dir="/tmp/out",
        log_path="/tmp/log.txt",
        max_steps=3,
    )
    crud.upsert_job_loss_point(test_db_session, job_id, group_index=0, step=1, loss=1.2)
    crud.upsert_job_loss_point(test_db_session, job_id, group_index=0, step=2, loss=0.9)

    response = client.get(f"/api/v1/train_jobs/{job_id}", headers=_auth_header())
    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 0
    assert body["data"]["job_id"] == job_id
    assert body["data"]["status"] == "pending"
    assert body["data"]["is_completed"] is False
    assert body["data"]["loss_data"]["steps"] == [1, 2]
    assert body["data"]["loss_data"]["values"] == [1.2, 0.9]
    assert body["data"]["duration"] >= 0


def test_get_train_job_status_not_found(client: TestClient):
    response = client.get("/api/v1/train_jobs/not-exist", headers=_auth_header())
    assert response.status_code == 404
    body = response.json()
    assert body["code"] == 40401
    assert body["message"] == "job not found"
