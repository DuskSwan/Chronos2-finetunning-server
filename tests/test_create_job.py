"""
Tests for fine-tuning job creation endpoint.
"""

import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.main import create_app
from app.db.models import Base
from app.db.session import get_db
from app.core.config import Settings


@pytest.fixture
def temp_base_dir():
    """Create a temporary base directory for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Clean up with a small delay to allow file handles to close
    import time
    time.sleep(0.1)
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
        save_request_artifacts=True,
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
        with patch("app.api.finetune.get_settings", return_value=test_settings):
            with patch("app.main.initialize_worker", lambda *_args, **_kwargs: None):
                with patch("app.core.paths.ensure_dir") as mock_ensure_dir:
                    # Mock ensure_dir to just return the path
                    mock_ensure_dir.side_effect = lambda path: path
                    app = create_app()
                    app.dependency_overrides[get_db] = get_test_db
                    with TestClient(app) as test_client:
                        yield test_client


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_create_finetune_job_success(client, test_settings):
    """Test successful fine-tuning job creation."""
    request_data = {
        "train_data_path": "/path/to/train.csv",
        "val_data_path": "/path/to/val.csv",
        "prediction_length": 96,
        "context_length": 512,
        "finetune_mode": "lora",
        "learning_rate": 0.0001,
        "num_steps": 1000,
        "batch_size": 32,
    }
    
    response = client.post("/v1/finetune/jobs", json=request_data)
    
    # Check response status
    assert response.status_code == 201
    
    # Check response body
    data = response.json()
    assert "job_id" in data
    assert "status" in data
    assert data["status"] == "queued"
    
    job_id = data["job_id"]
    
    # Verify job_id format (UUID)
    assert len(job_id) == 36  # UUID format


def test_create_finetune_job_missing_required_field(client):
    """Test job creation with missing required field."""
    request_data = {
        # Missing train_data_path
        "prediction_length": 96,
    }
    
    response = client.post("/v1/finetune/jobs", json=request_data)
    assert response.status_code == 422  # Pydantic validation error


def test_create_finetune_job_empty_train_path(client):
    """Test job creation with empty train_data_path."""
    request_data = {
        "train_data_path": "",
        "prediction_length": 96,
    }
    
    response = client.post("/v1/finetune/jobs", json=request_data)
    assert response.status_code == 422  # Pydantic validation error
    assert "train_data_path" in response.text


def test_create_finetune_job_invalid_finetune_mode(client):
    """Test job creation with invalid finetune_mode."""
    request_data = {
        "train_data_path": "/path/to/train.csv",
        "prediction_length": 96,
        "finetune_mode": "invalid_mode",
    }
    
    response = client.post("/v1/finetune/jobs", json=request_data)
    assert response.status_code == 422  # Pydantic validation error
    assert "finetune_mode" in response.text


def test_create_finetune_job_invalid_prediction_length(client):
    """Test job creation with invalid prediction_length."""
    request_data = {
        "train_data_path": "/path/to/train.csv",
        "prediction_length": 0,  # Invalid: must be positive
    }
    
    response = client.post("/v1/finetune/jobs", json=request_data)
    assert response.status_code == 422
    assert "prediction_length" in response.text


def test_create_finetune_job_creates_task_directory(client, test_settings):
    """Test that job creation creates task directory and request.json."""
    request_data = {
        "train_data_path": "/path/to/train.csv",
        "prediction_length": 96,
    }
    
    response = client.post("/v1/finetune/jobs", json=request_data)
    assert response.status_code == 201
    
    job_id = response.json()["job_id"]
    job_dir = Path(test_settings.artifacts_root) / job_id
    
    # Check if directory was created
    assert job_dir.exists()
    
    # Check if request.json was created
    request_json_file = job_dir / "request.json"
    assert request_json_file.exists()
    
    # Verify request.json content
    with open(request_json_file) as f:
        saved_request = json.load(f)
    
    assert saved_request["train_data_path"] == "/path/to/train.csv"
    assert saved_request["prediction_length"] == 96


def test_create_finetune_job_writes_to_database(client, test_db_session):
    """Test that job creation writes to database."""
    from app.db.crud import get_job_by_id
    
    request_data = {
        "train_data_path": "/path/to/train.csv",
        "prediction_length": 96,
    }
    
    response = client.post("/v1/finetune/jobs", json=request_data)
    assert response.status_code == 201
    
    job_id = response.json()["job_id"]
    
    # Query database using the test session
    job = get_job_by_id(test_db_session, job_id)
    assert job is not None
    assert job.id == job_id
    assert job.status == "queued"
    assert job.max_steps == 1000  # Default num_steps


def test_create_finetune_job_default_values(client, test_db_session):
    """Test job creation with default values."""
    request_data = {
        "train_data_path": "/path/to/train.csv",
        "prediction_length": 96,
    }
    
    response = client.post("/v1/finetune/jobs", json=request_data)
    assert response.status_code == 201
    
    # Query database to check defaults
    from app.db.crud import get_job_by_id
    
    job_id = response.json()["job_id"]
    job = get_job_by_id(test_db_session, job_id)
    
    # Check database defaults
    assert job is not None
    assert job.max_steps == 1000  # Default num_steps
    assert job.current_step == 0
    assert job.cancel_requested is False
