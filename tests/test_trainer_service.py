"""Chronos-2 训练服务的测试。

本模块使用 mock 进行单元测试，不进行真实的长时间训练。
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from app.db.session import SessionLocal, engine
from app.db.models import Base, FinetuneJob
from app.db.crud import create_job, get_job_by_id, update_job_progress
from app.callbacks.progress_callback import ProgressCallback
from app.services.trainer_service import train_chronos2, mock_train
from app.services.dataset_service import (
    prepare_training_data,
    prepare_validation_data,
    validate_required_columns,
)


@pytest.fixture
def temp_db():
    """创建临时数据库用于测试。"""
    import gc
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        test_db_url = f"sqlite:///{db_path}"
        
        # 创建表
        test_engine = create_engine(test_db_url, echo=False)
        Base.metadata.create_all(test_engine)
        
        # 创建会话
        TestSessionLocal = sessionmaker(bind=test_engine)
        db = TestSessionLocal()
        
        yield db, tmpdir
        
        # 确保连接关闭
        db.close()
        test_engine.dispose()
        
        # 强制垃圾回收以释放文件句柄
        gc.collect()


@pytest.fixture
def sample_csv_data(tmp_path):
    """创建示例 CSV 数据文件。"""
    csv_content = """item_id,timestamp,target
item1,2024-01-01,100.5
item1,2024-01-02,101.3
item1,2024-01-03,99.8
item2,2024-01-01,200.1
item2,2024-01-02,202.5
item2,2024-01-03,201.2
"""
    csv_path = tmp_path / "train.csv"
    csv_path.write_text(csv_content)
    return str(csv_path)


class TestDatasetService:
    """数据集服务的测试。"""

    def test_prepare_training_data_csv(self, sample_csv_data):
        """测试从 CSV 加载训练数据。"""
        data = prepare_training_data(sample_csv_data)
        
        assert "target" in data
        assert len(data["target"]) == 2  # 两个 item
        assert len(data["target"][0]) == 3  # 每个 item 三个数据点
        assert data["target"][0] == [100.5, 101.3, 99.8]
        assert data["target"][1] == [200.1, 202.5, 201.2]

    def test_prepare_validation_data_none(self):
        """测试验证数据为 None 的情况。"""
        result = prepare_validation_data(None)
        assert result is None

    def test_validate_required_columns_missing(self, sample_csv_data):
        """测试缺少必需列时的错误。"""
        import pandas as pd
        
        df = pd.read_csv(sample_csv_data)
        df_missing = df.drop("target", axis=1)
        
        with pytest.raises(ValueError, match="缺少必需列"):
            validate_required_columns(df_missing)


class TestProgressCallback:
    """进度回调的测试。"""

    def test_on_step_end(self, temp_db):
        """测试 on_step_end 回调。"""
        db, tmpdir = temp_db
        
        # 创建示例任务
        job_id = "test-job-1"
        create_job(
            db,
            job_id=job_id,
            status="running",
            request_json='{}',
            output_dir=tmpdir,
            log_path=str(Path(tmpdir) / "train.log"),
            max_steps=100,
        )
        
        # 创建回调
        callback = ProgressCallback(
            db_session=db,
            job_id=job_id,
            log_path=str(Path(tmpdir) / "train.log"),
            max_steps=100,
        )
        
        # 模拟步骤
        callback.on_step_end(step=10, loss=0.5)
        
        # 验证数据库更新
        job = get_job_by_id(db, job_id)
        assert job.current_step == 10
        assert job.last_loss == 0.5

    def test_callback_writes_log(self, temp_db):
        """测试回调写入日志文件。"""
        db, tmpdir = temp_db
        job_id = "test-job-2"
        log_path = str(Path(tmpdir) / "train.log")
        
        callback = ProgressCallback(
            db_session=db,
            job_id=job_id,
            log_path=log_path,
            max_steps=100,
        )
        
        callback._write_log("test message")
        
        # 验证日志文件
        assert Path(log_path).exists()
        content = Path(log_path).read_text(encoding="utf-8")
        assert "test message" in content


class TestTrainerService:
    """训练服务的测试。"""

    def test_mock_train(self, tmp_path):
        """测试模拟训练器。"""
        config = {"output_dir": str(tmp_path)}
        
        model_path = mock_train(config, steps=2)
        
        assert model_path == str(tmp_path / "finetuned-ckpt")

    @patch("app.services.trainer_service.prepare_training_data")
    @patch("app.services.trainer_service.prepare_validation_data")
    @patch("chronos.ChronosPipeline")
    def test_train_chronos2_success(
        self,
        mock_pipeline_class,
        mock_prepare_val,
        mock_prepare_train,
        temp_db,
        tmp_path,
    ):
        """测试成功的 Chronos-2 训练（使用 mock）。"""
        db, tmpdir = temp_db
        job_id = "test-job-3"
        output_dir = str(tmp_path)
        log_path = str(Path(tmpdir) / "train.log")
        
        # 创建任务记录
        create_job(
            db,
            job_id=job_id,
            status="running",
            request_json='{}',
            output_dir=output_dir,
            log_path=log_path,
            max_steps=100,
        )
        
        # Mock 数据准备
        mock_prepare_train.return_value = {
            "target": [[1, 2, 3, 4, 5]],
        }
        mock_prepare_val.return_value = None
        
        # Mock 模型
        mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        mock_fine_tuned = MagicMock()
        mock_pipeline.fit.return_value = mock_fine_tuned
        
        # 调用训练
        result = train_chronos2(
            db=db,
            job_id=job_id,
            train_data_path="/fake/path/train.csv",
            val_data_path=None,
            output_dir=output_dir,
            log_path=log_path,
            model_id="amazon/chronos-2",
            prediction_length=96,
            num_steps=10,
        )
        
        # 验证
        assert result.endswith("finetuned-ckpt")
        
        # 验证数据准备调用
        mock_prepare_train.assert_called_once()

    @patch("app.services.trainer_service.prepare_training_data")
    def test_train_chronos2_data_not_found(
        self,
        mock_prepare_train,
        temp_db,
        tmp_path,
    ):
        """测试数据文件不存在的情况。"""
        db, tmpdir = temp_db
        job_id = "test-job-4"
        
        # Mock 数据加载失败
        mock_prepare_train.side_effect = FileNotFoundError("数据文件不存在")
        
        create_job(
            db,
            job_id=job_id,
            status="running",
            request_json='{}',
            output_dir=str(tmp_path),
            log_path=str(Path(tmpdir) / "train.log"),
        )
        
        # 验证异常
        with pytest.raises(FileNotFoundError):
            train_chronos2(
                db=db,
                job_id=job_id,
                train_data_path="/fake/path/train.csv",
                val_data_path=None,
                output_dir=str(tmp_path),
                log_path=str(Path(tmpdir) / "train.log"),
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
