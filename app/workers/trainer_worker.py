"""后台训练 worker。

从队列中持续消费任务并进行训练。以线程方式后台运行。
"""

import json
import threading
from typing import Optional

from app.services.queue_service import get_job_queue
from app.services.job_service import (
    start_job_training,
    complete_job_training,
    fail_job_training,
)
from app.services.trainer_service import train_chronos2
from app.db.session import SessionLocal
from app.db.crud import get_job_by_id, mark_job_cancelled
from app.core.enums import JobStatus
from app.core.config import Settings
from app.callbacks.progress_callback import CancelledError


from loguru import logger


class TrainerWorker:
    """后台训练 worker，持续从队列消费任务。"""

    def __init__(self, settings: Settings, poll_interval: float = 1.0) -> None:
        """初始化 worker。

        Args:
            settings: 应用配置。
            poll_interval: 队列轮询间隔（秒）。
        """
        self.settings = settings
        self.poll_interval = poll_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.queue = get_job_queue()

    def start(self) -> None:
        """启动 worker 线程。"""
        if self._running:
            logger.warning("Worker 已在运行中")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("后台训练 Worker 已启动")

    def stop(self) -> None:
        """停止 worker 线程。"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("后台训练 Worker 已停止")

    def _run(self) -> None:
        """Worker 主循环。"""
        while self._running:
            try:
                # 从队列获取任务，超时 0.5 秒
                job_id = self.queue.dequeue(timeout=0.5)
                
                if job_id is None:
                    # 队列空，继续轮询
                    continue
                
                logger.info(f"开始处理任务: {job_id}")
                self._process_job(job_id)
                
            except Exception as e:
                logger.error(f"Worker 中发生错误: {e}", exc_info=True)

    def _process_job(self, job_id: str) -> None:
        """处理单个任务的完整工作流。

        Args:
            job_id: 任务 ID。
        """
        db = SessionLocal()
        
        try:
            # 1. 获取任务信息
            db.expire_all()
            job = get_job_by_id(db, job_id)
            if not job:
                logger.error(f"任务不存在: {job_id}")
                return

            # 若任务已取消，直接跳过
            if job.status == JobStatus.cancelled.value:
                logger.info(f"任务已取消，跳过处理: {job_id}")
                return

            # queued 状态下若已请求取消，直接标记为 cancelled
            if job.cancel_requested and job.status == JobStatus.queued.value:
                mark_job_cancelled(db, job_id)
                logger.info(f"任务已请求取消（queued），已标记为 cancelled: {job_id}")
                return
            
            # 2. 更新为运行状态
            start_job_training(db, job_id)
            logger.info(f"任务 {job_id} 已更新为 running")
            
            # 3. 解析请求信息
            request_data = json.loads(job.request_json)
            
            # 4. 准备训练配置
            train_config = {
                "db": db,
                "job_id": job_id,
                "train_data_path": request_data.get("train_data_path"),
                "val_data_path": request_data.get("val_data_path"),
                "output_dir": job.output_dir,
                "log_path": job.log_path,
                "prediction_length": request_data.get("prediction_length"),
                "context_length": request_data.get("context_length"),
                "finetune_mode": request_data.get("finetune_mode"),
                "learning_rate": request_data.get("learning_rate"),
                "num_steps": request_data.get("num_steps"),
                "batch_size": request_data.get("batch_size"),
                "logging_steps": request_data.get("logging_steps"),
                "finetuned_ckpt_name": request_data.get("finetuned_ckpt_name"),
                "device": request_data.get("device"),
                "selected_columns": request_data.get("selected_columns"),
            }

            # 5. 执行真实训练
            logger.info(f"开始真实训练任务: {job_id}")
            model_path = train_chronos2(**train_config)

            # 6. 标记为完成（若未取消）
            db.expire_all()
            job = get_job_by_id(db, job_id)
            if job and job.cancel_requested:
                mark_job_cancelled(db, job_id)
                logger.info(f"任务 {job_id} 在完成前被取消")
                return
            
            complete_job_training(db, job_id, model_path)
            logger.info(f"任务 {job_id} 已完成，模型路径: {model_path}")
            
        except CancelledError as e:
            logger.info(f"任务 {job_id} 被取消: {e}")
            mark_job_cancelled(db, job_id)
        except Exception as e:
            logger.error(f"任务 {job_id} 处理失败: {e}", exc_info=True)
            fail_job_training(db, job_id, str(e))
        finally:
            db.close()


# 全局 worker 实例
_global_worker: Optional[TrainerWorker] = None


def get_trainer_worker() -> TrainerWorker:
    """获取全局 worker 实例。

    Returns:
        TrainerWorker 实例。
    """
    global _global_worker
    if _global_worker is None:
        from app.core.config import get_settings
        settings = get_settings()
        _global_worker = TrainerWorker(settings)
    return _global_worker


def initialize_worker(settings: Settings) -> TrainerWorker:
    """初始化并启动全局 worker。

    Args:
        settings: 应用配置。

    Returns:
        启动后的 TrainerWorker 实例。
    """
    global _global_worker
    _global_worker = TrainerWorker(settings)
    _global_worker.start()
    return _global_worker
