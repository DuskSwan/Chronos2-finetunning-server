"""Chronos-2 训练过程中的进度回调。

在每个训练步骤后更新数据库中的进度信息（当前步数、最大步数、最后损失值）
和日志文件。
"""

import logging
from pathlib import Path
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.db.crud import update_job_progress, get_job_by_id


logger = logging.getLogger(__name__)


class CancelledError(RuntimeError):
    """训练过程中检测到取消请求时抛出。"""


class ProgressCallback:
    """Chronos-2 fit() 过程中的进度回调。

    在每个训练步骤后调用，用于更新数据库和日志。

    参数：
        db_session: SQLAlchemy 数据库会话。
        job_id: 任务 ID。
        log_path: 日志文件路径。
        max_steps: 总步数（可选）。
    """

    def __init__(
        self,
        db_session: Session,
        job_id: str,
        log_path: str,
        max_steps: Optional[int] = None,
    ) -> None:
        """初始化回调。

        Args:
            db_session: 数据库会话。
            job_id: 微调任务 ID。
            log_path: 日志文件路径。
            max_steps: 期望的最大步数（用于进度展示）。
        """
        self.db_session = db_session
        self.job_id = job_id
        self.log_path = Path(log_path)
        self.max_steps = max_steps
        self.current_step = 0
        
        # 确保日志目录存在
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def on_step_end(
        self,
        step: int,
        loss: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        """在每个训练步骤结束后调用。

        此方法由 Chronos-2 fit() 或其他训练循环调用，
        用于报告每步的指标。

        Args:
            step: 当前步数（从 0 或 1 开始）。
            loss: 当前步的损失值（可选）。
            **kwargs: 其他可能的指标字段。
        """
        self.check_cancel_requested()
        self.current_step = step

        try:
            # 更新数据库中的进度
            update_job_progress(
                db=self.db_session,
                job_id=self.job_id,
                current_step=step,
                max_steps=self.max_steps,
                last_loss=loss,
            )

            # 记录到日志文件
            self._log_progress(step, loss, **kwargs)

            logger.debug(
                f"任务 {self.job_id} 进度更新: 步骤 {step}, "
                f"损失 {loss if loss is not None else 'N/A'}"
            )
        except Exception as e:
            logger.error(
                f"回调更新失败 (任务 {self.job_id}, 步骤 {step}): {e}",
                exc_info=True,
            )

    def on_training_end(
        self,
        final_loss: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        """在训练结束时调用。

        Args:
            final_loss: 最终损失值（可选）。
            **kwargs: 其他可能的最终指标。
        """
        message = f"训练结束: 最后一步 {self.current_step}"
        if final_loss is not None:
            message += f", 最终损失 {final_loss:.6f}"
        
        logger.info(message)
        self._write_log(message)

    def on_training_start(self, **kwargs: Any) -> None:
        """在训练开始时调用。

        Args:
            **kwargs: 其他训练参数。
        """
        message = f"训练开始: 任务 {self.job_id}"
        logger.info(message)
        self._write_log(message)

    def on_exception(self, exception: Exception) -> None:
        """在训练过程中发生异常时调用。

        Args:
            exception: 捕获到的异常。
        """
        message = f"训练异常: {type(exception).__name__}: {str(exception)}"
        logger.error(message)
        self._write_log(message)

    def _log_progress(
        self,
        step: int,
        loss: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        """将进度信息写入日志文件。

        Args:
            step: 当前步。
            loss: 损失值。
            **kwargs: 其他指标。
        """
        parts = [f"[步骤 {step}]"]
        
        if loss is not None:
            parts.append(f"损失={loss:.6f}")
        
        for key, value in kwargs.items():
            if value is not None:
                parts.append(f"{key}={value}")
        
        message = ", ".join(parts)
        self._write_log(message)

    def _write_log(self, message: str) -> None:
        """将信息写入日志文件。

        Args:
            message: 要记录的信息。
        """
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"{message}\n")
        except Exception as e:
            logger.warning(f"无法写入日志文件 {self.log_path}: {e}")

    def check_cancel_requested(self) -> None:
        """检查是否请求取消任务，若请求则抛出取消异常。"""
        try:
            self.db_session.expire_all()
            job = get_job_by_id(self.db_session, self.job_id)
        except Exception:
            job = None

        if job and job.cancel_requested:
            message = "检测到取消请求，准备中止训练"
            self._write_log(message)
            raise CancelledError(message)
