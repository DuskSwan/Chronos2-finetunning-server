"""后台训练 worker。

从队列中持续消费任务并进行训练。以线程方式后台运行。
"""

import json
import logging
import threading
import time
from typing import Optional

from app.services.queue_service import get_job_queue
from app.services.job_service import (
    start_job_training,
    update_job_step,
    complete_job_training,
    fail_job_training,
)
from app.services.trainer_service import mock_train
from app.db.session import SessionLocal
from app.db.crud import get_job_by_id
from app.core.config import Settings


logger = logging.getLogger(__name__)


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
            job = get_job_by_id(db, job_id)
            if not job:
                logger.error(f"任务不存在: {job_id}")
                return
            
            # 2. 更新为运行状态
            start_job_training(db, job_id)
            logger.info(f"任务 {job_id} 已更新为 running")
            
            # 3. 解析请求信息
            request_data = json.loads(job.request_json)
            
            # 4. 准备训练配置
            train_config = {
                "output_dir": job.output_dir,
                "job_id": job_id,
                **request_data,
            }
            
            # 5. 执行假训练
            max_steps = 5  # 假训练总步数
            logger.info(f"开始模拟训练，总步数: {max_steps}")
            
            for step in range(1, max_steps + 1):
                if not self._running:
                    logger.info(f"任务 {job_id} 被中断")
                    break
                
                # 执行一步训练（耗时 0.2 ~ 0.5 秒）
                import random
                sleep_time = random.uniform(0.2, 0.5)
                time.sleep(sleep_time)
                
                # 模拟损失值
                loss = 1.0 - (step / max_steps) * 0.7 + random.uniform(-0.02, 0.02)
                loss = max(loss, 0.1)
                
                # 更新进度
                update_job_step(
                    db=db,
                    job_id=job_id,
                    current_step=step,
                    max_steps=max_steps,
                    last_loss=loss,
                )
                logger.info(f"任务 {job_id} 进度: {step}/{max_steps}, 损失: {loss:.4f}")
            
            # 6. 调用完整的假训练器（可选，也可以跳过）
            # model_path = mock_train(train_config, steps=max_steps)
            
            # 为演示目的，直接返回假的模型路径
            model_path = f"{job.output_dir}/finetuned-ckpt"
            
            # 7. 标记为完成
            complete_job_training(db, job_id, model_path)
            logger.info(f"任务 {job_id} 已完成，模型路径: {model_path}")
            
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
