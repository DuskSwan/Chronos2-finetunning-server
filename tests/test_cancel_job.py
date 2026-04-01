"""
Tests for fine-tuning job cancel endpoint and behavior.
"""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import create_app
from app.core.config import Settings
from app.db.models import Base
from app.db.session import get_db
from app.db import crud
from app.workers.trainer_worker import TrainerWorker


@pytest.fixture
def temp_dirs():
    """创建临时目录。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        db_dir = tmpdir / "db"
        artifacts_dir = tmpdir / "artifacts"
        logs_dir = tmpdir / "logs"

        db_dir.mkdir(exist_ok=True)
        artifacts_dir.mkdir(exist_ok=True)
        logs_dir.mkdir(exist_ok=True)

        yield {
            "db": db_dir,
            "artifacts": artifacts_dir,
            "logs": logs_dir,
            "root": tmpdir,
        }


@pytest.fixture
def test_settings(temp_dirs) -> Settings:
    """创建测试配置。"""
    return Settings(
        host="127.0.0.1",
        port=8000,
        sqlite_db_path=str(temp_dirs["db"] / "test.db"),
        artifacts_root=str(temp_dirs["artifacts"]),
        logs_root=str(temp_dirs["logs"]),
    )


@pytest.fixture
def isolated_app(test_settings: Settings, monkeypatch):
    """创建独立应用实例并隔离数据库与 SessionLocal。"""
    db_url = f"sqlite:///{Path(test_settings.sqlite_db_path).as_posix()}"
    engine = create_engine(
        db_url,
        connect_args={"check_same_thread": False, "timeout": 10},
    )
    Base.metadata.create_all(bind=engine)

    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def mock_get_settings():
        return test_settings

    monkeypatch.setattr("app.db.session.SessionLocal", TestSessionLocal)
    monkeypatch.setattr("app.core.config.get_settings", mock_get_settings)

    app = create_app()

    yield app, TestSessionLocal, engine

    engine.dispose()


@pytest.fixture
def client(isolated_app):
    """创建测试客户端。"""
    app, TestSessionLocal, _ = isolated_app

    def get_test_db():
        db = TestSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = get_test_db
    return TestClient(app)


def test_cancel_queued_job_success(client: TestClient, isolated_app):
    """queued 任务取消成功。"""
    _, TestSessionLocal, _ = isolated_app
    job_id = "cancel-queued-1"

    db = TestSessionLocal()
    crud.create_job(
        db=db,
        job_id=job_id,
        status="queued",
        request_json="{}",
        output_dir="/tmp/output",
        log_path="/tmp/output/train.log",
        max_steps=5,
    )
    db.close()

    response = client.post(f"/v1/finetune/jobs/{job_id}/cancel")
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == job_id
    assert data["status"] == "cancelled"
    assert data["cancel_requested"] is True

    db = TestSessionLocal()
    job = crud.get_job_by_id(db, job_id)
    assert job.status == "cancelled"
    assert job.cancel_requested is True
    assert job.finished_at is not None
    db.close()


def test_cancel_running_job_eventually_cancelled(
    client: TestClient,
    isolated_app,
    monkeypatch,
    test_settings: Settings,
):
    """running 任务请求取消后最终状态为 cancelled。"""
    _, TestSessionLocal, _ = isolated_app
    job_id = "cancel-running-1"

    db = TestSessionLocal()
    crud.create_job(
        db=db,
        job_id=job_id,
        status="queued",
        request_json="{}",
        output_dir="/tmp/output",
        log_path="/tmp/output/train.log",
        max_steps=5,
    )
    crud.update_job_status(
        db=db,
        job_id=job_id,
        status="running",
        started_at=datetime.now(timezone.utc),
    )
    db.close()

    response = client.post(f"/v1/finetune/jobs/{job_id}/cancel")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert data["cancel_requested"] is True

    import random

    monkeypatch.setattr("app.workers.trainer_worker.time.sleep", lambda *_: None)
    monkeypatch.setattr(random, "uniform", lambda *_: 0)

    worker = TrainerWorker(test_settings)
    worker._process_job(job_id)

    db = TestSessionLocal()
    job = crud.get_job_by_id(db, job_id)
    assert job.status == "cancelled"
    db.close()


def test_cancel_completed_job_returns_conflict(client: TestClient, isolated_app):
    """已完成任务取消时返回合理响应。"""
    _, TestSessionLocal, _ = isolated_app
    job_id = "cancel-completed-1"
    output_dir = "/tmp/output"

    db = TestSessionLocal()
    crud.create_job(
        db=db,
        job_id=job_id,
        status="queued",
        request_json="{}",
        output_dir=output_dir,
        log_path="/tmp/output/train.log",
        max_steps=5,
    )
    crud.mark_job_completed(
        db=db,
        job_id=job_id,
        model_path=f"{output_dir}/finetuned-ckpt",
        finished_at=datetime.now(timezone.utc),
    )
    db.close()

    response = client.post(f"/v1/finetune/jobs/{job_id}/cancel")
    assert response.status_code == 409


def test_cancelled_job_not_marked_completed(client: TestClient, isolated_app, test_settings: Settings):
    """取消后的任务不会被写成 completed。"""
    _, TestSessionLocal, _ = isolated_app
    job_id = "cancelled-not-completed-1"

    db = TestSessionLocal()
    crud.create_job(
        db=db,
        job_id=job_id,
        status="queued",
        request_json="{}",
        output_dir="/tmp/output",
        log_path="/tmp/output/train.log",
        max_steps=5,
    )
    db.close()

    response = client.post(f"/v1/finetune/jobs/{job_id}/cancel")
    assert response.status_code == 200

    worker = TrainerWorker(test_settings)
    worker._process_job(job_id)

    db = TestSessionLocal()
    job = crud.get_job_by_id(db, job_id)
    assert job.status == "cancelled"
    db.close()
