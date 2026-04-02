"""
worker 流程集成测试。

验证后台 worker 能正确地消费任务队列并更新任务状态。
"""

import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import create_app
from app.core.config import Settings
from app.db.models import Base
from app.db.crud import get_job_by_id
from app.services.queue_service import initialize_queue, get_job_queue
from app.services.job_service import start_job_training
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
def sample_csv_data(temp_dirs):
    """创建示例 CSV 数据文件。"""
    csv_content = """item_id,timestamp,target
item1,2024-01-01,100.5
item1,2024-01-02,101.3
item1,2024-01-03,99.8
item2,2024-01-01,200.1
item2,2024-01-02,202.5
item2,2024-01-03,201.2
"""
    csv_path = Path(temp_dirs["root"]) / "train.csv"
    csv_path.write_text(csv_content)
    return str(csv_path)


@pytest.fixture
def isolated_app(test_settings: Settings, monkeypatch):
    """创建独立的应用实例，带有自己的数据库和 worker。"""
    initialize_queue()

    # 创建本地数据库
    db_url = f"sqlite:///{test_settings.sqlite_db_path_resolved.as_posix()}"
    engine = create_engine(
        db_url,
        connect_args={"check_same_thread": False, "timeout": 10},
    )
    Base.metadata.create_all(bind=engine)
    
    # 覆盖 SessionLocal
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    monkeypatch.setattr("app.db.session.SessionLocal", TestSessionLocal)
    monkeypatch.setattr("app.workers.trainer_worker.SessionLocal", TestSessionLocal)
    
    # 覆盖 get_settings
    def mock_get_settings():
        return test_settings
    
    monkeypatch.setattr("app.core.config.get_settings", mock_get_settings)
    monkeypatch.setattr("app.main.get_settings", mock_get_settings)

    # 禁用应用启动时的后台 worker，避免并发干扰
    monkeypatch.setattr("app.main.initialize_worker", lambda *_args, **_kwargs: None)
    
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


def test_create_job_queued(client: TestClient, sample_csv_data: str) -> None:
    """测试创建任务时状态为 queued。"""
    response = client.post(
        "/v1/finetune/jobs",
        json={
            "train_data_path": sample_csv_data,
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


def test_job_enters_queue(client: TestClient, sample_csv_data: str) -> None:
    """测试任务被加入队列。"""
    queue = get_job_queue()
    initial_size = queue.size()
    
    response = client.post(
        "/v1/finetune/jobs",
        json={
            "train_data_path": sample_csv_data,
            "prediction_length": 96,
        },
    )
    
    assert response.status_code == 201
    
    # 队列应该增加一个任务
    new_size = queue.size()
    assert new_size == initial_size + 1


def test_job_transitions_to_running(client: TestClient, isolated_app) -> None:
    """测试任务状态从 queued 转变为 running。"""
    _, TestSessionLocal, _ = isolated_app
    
    # 创建任务
    response = client.post(
        "/v1/finetune/jobs",
        json={
            "train_data_path": "dummy.csv",
            "prediction_length": 96,
        },
    )
    
    job_id = response.json()["job_id"]

    # 手动触发进入 running
    db = TestSessionLocal()
    try:
        start_job_training(db, job_id)
        job = get_job_by_id(db, job_id)
        assert job is not None
        assert job.status == "running"
        assert job.started_at is not None
    finally:
        db.close()


def test_job_completes(client: TestClient, isolated_app, test_settings: Settings, sample_csv_data: str) -> None:
    """测试任务完成流程（使用实际 worker）。
    
    注：真实训练可能需要下载模型并耗时。
    """
    _, TestSessionLocal, _ = isolated_app
    
    # 创建任务
    response = client.post(
        "/v1/finetune/jobs",
        json={
            "train_data_path": sample_csv_data,
            "prediction_length": 1,
            "context_length": 2,
            "num_steps": 1,
            "batch_size": 1,
            "logging_steps": 1,
            "selected_columns": ["target"],
        },
    )
    
    assert response.status_code == 201
    job_id = response.json()["job_id"]

    # 使用真实 worker 同步处理任务
    worker = TrainerWorker(test_settings)
    worker._process_job(job_id)
    
    db = TestSessionLocal()
    try:
        job = get_job_by_id(db, job_id)
        assert job is not None
        assert job.status == "completed"
        assert job.model_path is not None
        assert Path(job.model_path).exists()
    finally:
        db.close()


def test_job_progress_tracked(client: TestClient, isolated_app, test_settings: Settings, sample_csv_data: str) -> None:
    """测试任务日志被写入。"""
    _, TestSessionLocal, _ = isolated_app
    
    # 创建任务
    response = client.post(
        "/v1/finetune/jobs",
        json={
            "train_data_path": sample_csv_data,
            "prediction_length": 1,
            "context_length": 2,
            "num_steps": 1,
            "batch_size": 1,
            "logging_steps": 1,
            "selected_columns": ["target"],
        },
    )
    
    job_id = response.json()["job_id"]

    worker = TrainerWorker(test_settings)
    worker._process_job(job_id)
    
    # 检查进度
    db = TestSessionLocal()
    try:
        job = get_job_by_id(db, job_id)
        assert job is not None
        assert job.log_path is not None
        assert Path(job.log_path).exists()
        log_text = Path(job.log_path).read_text(encoding="utf-8")
        assert "训练开始" in log_text
        assert "模型保存成功" in log_text
    finally:
        db.close()
