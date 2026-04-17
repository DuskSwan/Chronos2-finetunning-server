"""
Tests for fine-tuning job query endpoints.
"""

import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import create_app
from app.core.config import Settings
from app.db.models import Base
from app.db.session import get_db
from app.db import crud


@pytest.fixture
def temp_base_dir():
    """Create a temporary base directory for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_settings(temp_base_dir):
    """Create test settings with temporary directories."""
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
    )


@pytest.fixture
def test_db_session(test_settings):
    """Create a test database and session."""
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
    """Create a test client with mocked dependencies."""
    def get_test_db():
        yield test_db_session

    with patch("app.main.get_settings", return_value=test_settings):
        with patch("app.main.initialize_worker", lambda *_args, **_kwargs: None):
            app = create_app()
            app.dependency_overrides[get_db] = get_test_db
            return TestClient(app)


def test_get_job_detail_success(client: TestClient, test_db_session):
    """Test querying an existing job detail."""
    job_id = "job-detail-1"
    output_dir = "/tmp/output/job-detail-1"
    log_path = "/tmp/output/job-detail-1/train.log"

    crud.create_job(
        db=test_db_session,
        job_id=job_id,
        status="queued",
        request_json="{}",
        output_dir=output_dir,
        log_path=log_path,
        max_steps=10,
    )

    started_at = datetime.now(timezone.utc)
    crud.update_job_status(
        db=test_db_session,
        job_id=job_id,
        status="running",
        started_at=started_at,
    )
    crud.update_job_progress(
        db=test_db_session,
        job_id=job_id,
        current_step=3,
        max_steps=10,
        last_loss=0.5,
    )

    response = client.get(f"/v1/finetune/jobs/{job_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == job_id
    assert data["status"] == "running"
    assert data["progress"]["current_step"] == 3
    assert data["progress"]["max_steps"] == 10
    assert data["progress"]["last_loss"] == pytest.approx(0.5)
    assert data["log_path"] == log_path


def test_get_job_detail_not_found(client: TestClient):
    """Test querying a non-existent job detail."""
    response = client.get("/v1/finetune/jobs/not-exist")
    assert response.status_code == 404


def test_get_job_result_completed(client: TestClient, test_db_session):
    """Test querying result of a completed job."""
    job_id = "job-result-1"
    output_dir = "/tmp/output/job-result-1"
    log_path = "/tmp/output/job-result-1/train.log"

    crud.create_job(
        db=test_db_session,
        job_id=job_id,
        status="queued",
        request_json="{}",
        output_dir=output_dir,
        log_path=log_path,
        max_steps=5,
    )

    crud.mark_job_completed(
        db=test_db_session,
        job_id=job_id,
        model_paths=[f"{output_dir}/finetuned-ckpt"],
        finished_at=datetime.now(timezone.utc),
    )

    response = client.get(f"/v1/finetune/jobs/{job_id}/result")
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == job_id
    assert data["status"] == "completed"
    assert data["output_dir"] == output_dir
    assert data["model_paths"] == [f"{output_dir}/finetuned-ckpt"]
    assert data["metrics"] == {
        "loss_steps": [[]],
        "loss_values": [[]],
        "loss_curve": [[]],
    }


def test_get_job_result_with_loss_curve(client: TestClient, test_db_session):
    """Test querying completed job result with loss curve data."""
    job_id = "job-result-loss-1"
    output_dir = "/tmp/output/job-result-loss-1"
    log_path = "/tmp/output/job-result-loss-1/train.log"

    crud.create_job(
        db=test_db_session,
        job_id=job_id,
        status="queued",
        request_json="{}",
        output_dir=output_dir,
        log_path=log_path,
        max_steps=5,
    )
    crud.upsert_job_loss_point(test_db_session, job_id, group_index=0, step=1, loss=0.9)
    crud.upsert_job_loss_point(test_db_session, job_id, group_index=0, step=2, loss=0.7)
    crud.upsert_job_loss_point(test_db_session, job_id, group_index=0, step=3, loss=0.5)
    crud.mark_job_completed(
        db=test_db_session,
        job_id=job_id,
        model_paths=[f"{output_dir}/finetuned-ckpt"],
        finished_at=datetime.now(timezone.utc),
    )

    response = client.get(f"/v1/finetune/jobs/{job_id}/result")
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == job_id
    assert data["metrics"]["loss_steps"] == [[1, 2, 3]]
    assert data["metrics"]["loss_values"] == [[0.9, 0.7, 0.5]]
    assert data["metrics"]["loss_curve"] == [[
        {"step": 1, "loss": pytest.approx(0.9)},
        {"step": 2, "loss": pytest.approx(0.7)},
        {"step": 3, "loss": pytest.approx(0.5)},
    ]]


def test_get_job_result_with_multi_group_loss_curves(client: TestClient, test_db_session):
    """Test querying completed job result with multi-group loss curves."""
    job_id = "job-result-loss-2"
    output_dir = "/tmp/output/job-result-loss-2"
    log_path = "/tmp/output/job-result-loss-2/train.log"

    crud.create_job(
        db=test_db_session,
        job_id=job_id,
        status="queued",
        request_json="{}",
        output_dir=output_dir,
        log_path=log_path,
        max_steps=6,
    )
    crud.upsert_job_loss_point(test_db_session, job_id, group_index=0, step=1, loss=0.9)
    crud.upsert_job_loss_point(test_db_session, job_id, group_index=0, step=2, loss=0.8)
    crud.upsert_job_loss_point(test_db_session, job_id, group_index=1, step=1, loss=0.6)
    crud.upsert_job_loss_point(test_db_session, job_id, group_index=1, step=2, loss=0.4)
    crud.mark_job_completed(
        db=test_db_session,
        job_id=job_id,
        model_paths=[
            f"{output_dir}/finetuned-ckpt_target_a",
            f"{output_dir}/finetuned-ckpt_target_b",
        ],
        finished_at=datetime.now(timezone.utc),
    )

    response = client.get(f"/v1/finetune/jobs/{job_id}/result")
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == job_id
    assert data["metrics"]["loss_steps"] == [[1, 2], [1, 2]]
    assert data["metrics"]["loss_values"] == [[0.9, 0.8], [0.6, 0.4]]
    assert data["metrics"]["loss_curve"] == [
        [
            {"step": 1, "loss": pytest.approx(0.9)},
            {"step": 2, "loss": pytest.approx(0.8)},
        ],
        [
            {"step": 1, "loss": pytest.approx(0.6)},
            {"step": 2, "loss": pytest.approx(0.4)},
        ],
    ]


def test_get_job_result_not_completed(client: TestClient, test_db_session):
    """Test querying result of a non-completed job."""
    job_id = "job-result-2"
    output_dir = "/tmp/output/job-result-2"
    log_path = "/tmp/output/job-result-2/train.log"

    crud.create_job(
        db=test_db_session,
        job_id=job_id,
        status="queued",
        request_json="{}",
        output_dir=output_dir,
        log_path=log_path,
        max_steps=5,
    )

    response = client.get(f"/v1/finetune/jobs/{job_id}/result")
    assert response.status_code == 409


def test_get_job_logs_success(client: TestClient, test_db_session, temp_base_dir):
    """Test querying job logs."""
    job_id = "job-logs-1"
    output_dir = str(temp_base_dir / "output" / job_id)
    log_path = temp_base_dir / "logs" / f"{job_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("line1\nline2\nline3\n", encoding="utf-8")

    crud.create_job(
        db=test_db_session,
        job_id=job_id,
        status="running",
        request_json="{}",
        output_dir=output_dir,
        log_path=str(log_path),
        max_steps=5,
    )

    response = client.get(f"/v1/finetune/jobs/{job_id}/logs")
    assert response.status_code == 200
    assert "line1" in response.text
    assert "line2" in response.text
    assert "line3" in response.text


def test_calculate_correlation_matrix(client: TestClient, temp_base_dir) -> None:
    """测试相关性矩阵接口。"""
    csv_path = temp_base_dir / "test_data.csv"
    csv_path.write_text("a,b,c\n1,2,4\n2,3,5\n3,4,6\n", encoding="utf-8")
    
    response = client.post(
        "/v1/tools/correlation",
        json={"csv_path": str(csv_path), "columns": ["a", "c"]},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["correlation_matrix"]["a"]["a"] == pytest.approx(1.0)
    assert data["correlation_matrix"]["c"]["c"] == pytest.approx(1.0)
    assert data["correlation_matrix"]["a"]["c"] == pytest.approx(1.0)


def test_calculate_correlation_matrix_spearman(client: TestClient, temp_base_dir) -> None:
    """测试 Spearman 相关性矩阵接口。"""
    csv_path = temp_base_dir / "test_data_spearman.csv"
    csv_path.write_text("a,b,c\n1,2,4\n2,3,5\n3,4,6\n", encoding="utf-8")
    
    response = client.post(
        "/v1/tools/correlation",
        json={"csv_path": str(csv_path), "columns": ["a", "c"], "method": "spearman"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["correlation_matrix"]["a"]["a"] == pytest.approx(1.0)
    assert data["correlation_matrix"]["c"]["c"] == pytest.approx(1.0)
    assert data["correlation_matrix"]["a"]["c"] == pytest.approx(1.0)


def test_calculate_correlation_matrix_invalid_method(client: TestClient, temp_base_dir) -> None:
    """测试无效相关性计算方法。"""
    csv_path = temp_base_dir / "test_data_invalid.csv"
    csv_path.write_text("a,b,c\n1,2,4\n2,3,5\n3,4,6\n", encoding="utf-8")
    
    response = client.post(
        "/v1/tools/correlation",
        json={"csv_path": str(csv_path), "columns": ["a", "c"], "method": "invalid"},
    )

    assert response.status_code == 422  # Validation error
