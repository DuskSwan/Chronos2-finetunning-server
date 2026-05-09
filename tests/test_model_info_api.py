"""Tests for /api/model/info endpoint."""

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
    return Settings(
        host="127.0.0.1",
        port=8000,
        sqlite_db_path=str(temp_base_dir / "test.db"),
        artifacts_root=str(temp_base_dir / "artifacts"),
        logs_root=str(temp_base_dir / "logs"),
        release_path=str(temp_base_dir / "release"),
        api_bearer_token="spec-token",
    )


@pytest.fixture
def test_db_session(test_settings):
    db_url = f"sqlite:///{Path(test_settings.sqlite_db_path).as_posix()}"
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


def test_model_info_success(client: TestClient, temp_base_dir: Path):
    model_root = temp_base_dir / "release" / "models" / "u1" / "v1" / "job1"
    model_root.mkdir(parents=True, exist_ok=True)
    (model_root / "metadata.json").write_text(
        (
            '{"selected_groups":[{"target":"value1","covariates":["value2"]}],'
            '"prediction_length":3,"context_length":8}'
        ),
        encoding="utf-8",
    )
    response = client.get(
        "/api/model/info",
        headers={"Authorization": "Bearer spec-token"},
        params={"model_path": str(model_root.resolve())},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 0
    assert body["data"]["targets"] == ["value1"]
    assert body["data"]["prediction_length"] == 3
    assert body["data"]["context_length"] == 8

