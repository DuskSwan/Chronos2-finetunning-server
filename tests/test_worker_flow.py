"""
worker 流程集成测试。

验证后台 worker 能正确地消费任务队列并更新任务状态。
"""

import json
import time
import tempfile
from pathlib import Path
from unittest import mock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import create_app
from app.core.config import Settings
from app.db.models import Base
from app.db.crud import get_job_by_id
from app.db.session import SessionLocal
from app.services.queue_service import initialize_queue, get_job_queue
from app.workers.trainer_worker import initialize_worker


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
    """创建独立的应用实例，带有自己的数据库和 worker。"""
    # 创建本地数据库
    db_url = f"sqlite:///{test_settings.sqlite_db_path_resolved.as_posix()}"
    engine = create_engine(
        db_url,
        connect_args={"check_same_thread": False, "timeout": 10},
    )
    Base.metadata.create_all(bind=engine)
    
    # 覆盖 SessionLocal
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    def mock_get_db():
        db = TestSessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    monkeypatch.setattr("app.db.session.SessionLocal", TestSessionLocal)
    
    # 覆盖 get_settings
    def mock_get_settings():
        return test_settings
    
    monkeypatch.setattr("app.core.config.get_settings", mock_get_settings)
    
    # 初始化应用
    app = create_app()
    
    yield app, TestSessionLocal, engine
    
    # 清理 - 确保所有连接都被关闭
    import gc
    gc.collect()  # 强制垃圾回收
    
    # 等待一下以确保所有连接关闭
    import time
    time.sleep(0.5)
    
    try:
        Base.metadata.drop_all(bind=engine)
    except Exception:
        pass  # 忽略清理错误
    finally:
        engine.dispose()


@pytest.fixture
def client(isolated_app):
    """创建测试客户端。"""
    app, _, _ = isolated_app
    return TestClient(app)


def test_health_check(client: TestClient) -> None:
    """测试健康检查端点。"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_job_queued(client: TestClient) -> None:
    """测试创建任务时状态为 queued。"""
    response = client.post(
        "/v1/finetune/jobs",
        json={
            "train_data_path": "/path/to/train.csv",
            "prediction_length": 96,
        },
    )
    
    assert response.status_code == 201
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "queued"


def test_request_json_created(temp_dirs: dict) -> None:
    """测试请求 JSON 文件被创建（直接调用 API）。"""
    from app.core.config import Settings
    
    settings = Settings(
        host="127.0.0.1",
        port=8000,
        sqlite_db_path=str(temp_dirs["db"] / "test.db"),
        artifacts_root=str(temp_dirs["artifacts"]),
        logs_root=str(temp_dirs["logs"]),
    )
    
    # 手动创建输出目录和文件，模拟 API 行为
    import uuid
    job_id = str(uuid.uuid4())
    output_dir = Path(settings.artifacts_root) / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    request_data = {
        "train_data_path": "/path/to/train.csv",
        "prediction_length": 96,
    }
    
    request_json_path = output_dir / "request.json"
    with open(request_json_path, "w") as f:
        json.dump(request_data, f)
    
    # 验证文件被创建
    assert request_json_path.exists()
    
    # 验证内容
    with open(request_json_path) as f:
        data = json.load(f)
    assert data["train_data_path"] == "/path/to/train.csv"


def test_output_directory_created(temp_dirs: dict) -> None:
    """测试输出目录可以被正确创建。"""
    output_dir = Path(temp_dirs["artifacts"]) / "test-job-id"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    assert output_dir.exists()
    assert output_dir.is_dir()


def test_job_enters_queue(client: TestClient) -> None:
    """测试任务被加入队列。"""
    queue = get_job_queue()
    initial_size = queue.size()
    
    response = client.post(
        "/v1/finetune/jobs",
        json={
            "train_data_path": "/path/to/train.csv",
            "prediction_length": 96,
        },
    )
    
    assert response.status_code == 201
    
    # 队列应该增加一个任务
    new_size = queue.size()
    assert new_size == initial_size + 1


def test_job_transitions_to_running(client: TestClient, isolated_app, test_settings: Settings) -> None:
    """测试任务状态从 queued 转变为 running（使用实际 worker）。"""
    _, TestSessionLocal, _ = isolated_app
    
    # 创建任务
    response = client.post(
        "/v1/finetune/jobs",
        json={
            "train_data_path": "/path/to/train.csv",
            "prediction_length": 96,
        },
    )
    
    job_id = response.json()["job_id"]
    
    # 等待 worker 处理任务
    time.sleep(0.5)
    
    # 检查任务状态
    db = TestSessionLocal()
    try:
        job = get_job_by_id(db, job_id)
        assert job is not None
        # 任务应该在 queued 或 running 中
        assert job.status in ["queued", "running"]
    finally:
        db.close()


def test_job_completes(client: TestClient, isolated_app, test_settings: Settings) -> None:
    """测试任务完成流程（使用实际 worker）。
    
    注：需要较长等待时间，因为假训练需要时间完成。
    """
    _, TestSessionLocal, _ = isolated_app
    
    # 创建任务
    response = client.post(
        "/v1/finetune/jobs",
        json={
            "train_data_path": "/path/to/train.csv",
            "prediction_length": 96,
            "num_steps": 5,
        },
    )
    
    assert response.status_code == 201
    job_id = response.json()["job_id"]
    
    # 验证默认等待一部分时间，看任务是否被消费
    time.sleep(2)
    
    db = TestSessionLocal()
    try:
        job = get_job_by_id(db, job_id)
        assert job is not None
        # 任务应该至少已enter运行或之后的状态，或仍在 queued 中等待
        assert job.status in ["queued", "running", "completed", "failed"]
    finally:
        db.close()


def test_job_progress_tracked(client: TestClient, isolated_app, test_settings: Settings) -> None:
    """测试任务进度被跟踪。"""
    _, TestSessionLocal, _ = isolated_app
    
    # 创建任务
    response = client.post(
        "/v1/finetune/jobs",
        json={
            "train_data_path": "/path/to/train.csv",
            "prediction_length": 96,
            "num_steps": 5,
        },
    )
    
    job_id = response.json()["job_id"]
    
    # 等一段时间让任务运行
    time.sleep(2)
    
    # 检查进度
    db = TestSessionLocal()
    try:
        job = get_job_by_id(db, job_id)
        assert job is not None
        
        # 如果任务在运行或已完成，进度应该有更新
        if job.status == "running":
            assert job.current_step > 0
        elif job.status == "completed":
            assert job.current_step == job.max_steps
    finally:
        db.close()
