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


def test_infer_model_success_multi_target(client: TestClient):
    repo_root = Path(__file__).resolve().parents[1]
    csv_path = repo_root / "mock_train_data.csv"
    assert csv_path.exists()

    model_root = repo_root / "release" / "models" / "user_10001" / "v1.0.0" / "job_x"
    model_a = model_root / "finetuned-ckpt_value1"
    model_b = model_root / "finetuned-ckpt_value2"
    model_a.mkdir(parents=True, exist_ok=True)
    model_b.mkdir(parents=True, exist_ok=True)

    def fake_loader(path, device="cpu"):
        _ = device
        p = Path(path)
        if p.name.endswith("value1"):
            return _FakePipeline([1.0, 2.0, 3.0])
        return _FakePipeline([4.0, 5.0, 6.0])

    with patch("app.services.inference_service.load_local_model", side_effect=fake_loader):
        response = client.post(
            "/api/model/infer",
            headers=_auth_header(),
            json={
                "model_path": str(model_root.resolve()),
                "cov_group": [
                    {"target": "value1", "covariates": ["value2", "value3"]},
                    {"target": "value2", "covariates": ["value1", "value4"]},
                ],
                "prediction_length": 3,
                "csv_path": str(csv_path.resolve()),
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 0
    assert body["message"] == "success"
    assert body["data"]["predictions"] == [
        {"target": "value1", "prediction": [1.0, 2.0, 3.0]},
        {"target": "value2", "prediction": [4.0, 5.0, 6.0]},
    ]


def test_infer_model_missing_target_model(client: TestClient):
    repo_root = Path(__file__).resolve().parents[1]
    csv_path = repo_root / "mock_train_data.csv"
    model_root = repo_root / "release" / "models" / "user_10001" / "v1.0.0" / "job_y"
    model_root.mkdir(parents=True, exist_ok=True)
    (model_root / "finetuned-ckpt_value2").mkdir(parents=True, exist_ok=True)

    response = client.post(
        "/api/model/infer",
        headers=_auth_header(),
        json={
            "model_path": str(model_root.resolve()),
            "cov_group": [{"target": "value1", "covariates": ["value2"]}],
            "prediction_length": 3,
            "csv_path": str(csv_path.resolve()),
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 500
    assert body["message"] == "model for target 'value1' not found"
    assert body["data"] is None
