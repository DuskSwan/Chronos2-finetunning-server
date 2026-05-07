"""Tests for /api/model/infer compatibility endpoint."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import Settings
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
        with patch("app.core.auth.get_settings", return_value=test_settings):
            with patch("app.main.initialize_worker", lambda *_args, **_kwargs: None):
                app = create_app()
                app.dependency_overrides[get_db] = get_test_db
                with TestClient(app) as test_client:
                    yield test_client


def _auth_header() -> dict[str, str]:
    return {"Authorization": "Bearer spec-token"}


def test_infer_model_contract_not_implemented(client: TestClient, temp_base_dir: Path):
    csv_path = temp_base_dir / "infer.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")
    model_path = (temp_base_dir / "release" / "mock_model").resolve()
    model_path.mkdir(parents=True, exist_ok=True)

    response = client.post(
        "/api/model/infer",
        headers=_auth_header(),
        json={
            "model_path": str(model_path),
            "cov_group": [{"target": "a", "covariates": ["b"]}],
            "prediction_length": 16,
            "csv_path": str(csv_path),
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 501
    assert body["message"] == "inference service not implemented yet"
    assert body["data"] is None


def test_infer_model_invalid_request_shape(client: TestClient, temp_base_dir: Path):
    response = client.post(
        "/api/model/infer",
        headers=_auth_header(),
        json={
            "model_path": str((temp_base_dir / "release" / "mock_model").resolve()),
            "cov_group": [],
            "prediction_length": 16,
            "csv_path": str((temp_base_dir / "infer.csv").resolve()),
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 500
    assert "cov_group cannot be empty" in body["message"]
    assert body["data"] is None
