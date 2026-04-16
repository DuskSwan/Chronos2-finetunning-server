"""Chronos-2 训练过程中的进度回调。

在每个训练步骤后更新数据库中的进度信息（当前步数、最大步数、最后损失值）
和日志文件。
"""


from pathlib import Path
from typing import Any, Optional

from transformers.trainer_callback import TrainerCallback

from sqlalchemy.orm import Session

from app.db.crud import update_job_progress, get_job_by_id, upsert_job_loss_point


from loguru import logger


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
        self.last_loss: Optional[float] = None
        self.active_group_index = 0
        self.total_groups = 0
        self.active_group_target = ""
        self.active_group_max_steps = 0
        self.active_group_base_step = 0
        self.active_group_summary: dict[str, Any] = {}
        self.group_summaries: list[dict[str, Any]] = []

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
        if loss is not None:
            self.last_loss = loss

        try:
            # 更新数据库中的进度
            update_job_progress(
                db=self.db_session,
                job_id=self.job_id,
                current_step=step,
                max_steps=self.max_steps,
                last_loss=loss if loss is not None else self.last_loss,
            )
            if loss is not None:
                upsert_job_loss_point(
                    db=self.db_session,
                    job_id=self.job_id,
                    step=step,
                    loss=loss,
                )

            # 记录到日志文件
            self._log_progress(step, loss, **kwargs)

            logger.debug(
                f"任务 {self.job_id} 进度更新: 步骤 {step}, "
                f"损失 {self._format_loss(loss if loss is not None else self.last_loss)}"
            )
        except Exception as e:
            logger.error(
                f"回调更新失败 (任务 {self.job_id}, 步骤 {step}): {e}",
                exc_info=True,
            )

    def on_training_start(self, **kwargs: Any) -> None:
        """在训练开始时调用。

        Args:
            **kwargs: 其他训练参数。
        """
        message = f"训练开始: 任务 {self.job_id}"
        logger.info(message)
        self._write_log(message)

    def on_group_start(
        self,
        group_index: int,
        total_groups: int,
        target: str,
        group_max_steps: int,
    ) -> None:
        """记录一组训练的开始信息。"""
        self.active_group_index = group_index
        self.total_groups = total_groups
        self.active_group_target = target
        self.active_group_max_steps = group_max_steps
        self.active_group_base_step = group_index * group_max_steps
        self.active_group_summary = {}

        message = (
            f"开始第 {group_index + 1}/{total_groups} 组训练: "
            f"target={target}, 计划步数={group_max_steps}"
        )
        logger.info(message)
        self._write_log(message)

    def on_trainer_log(
        self,
        step: int,
        max_steps: Optional[int] = None,
        logs: Optional[dict[str, Any]] = None,
    ) -> None:
        """处理 HuggingFace Trainer 的日志事件。"""
        logs = logs or {}
        group_max_steps = max_steps or self.active_group_max_steps or step
        overall_step = min(
            self.max_steps or (self.active_group_base_step + step),
            self.active_group_base_step + step,
        )

        loss = self._coerce_float(logs.get("loss"))
        if loss is not None:
            self.last_loss = loss

        tracked_summary_keys = (
            "train_loss",
            "train_runtime",
            "train_steps_per_second",
            "train_samples_per_second",
        )
        for key in tracked_summary_keys:
            value = self._coerce_float(logs.get(key))
            if value is not None:
                self.active_group_summary[key] = value

        progress_metrics: dict[str, Any] = {}
        for key in ("learning_rate", "grad_norm"):
            value = self._coerce_float(logs.get(key))
            if value is not None:
                progress_metrics[key] = value

        if loss is not None:
            self.on_step_end(overall_step, loss=loss, **progress_metrics)
            return

        self.current_step = overall_step
        update_job_progress(
            db=self.db_session,
            job_id=self.job_id,
            current_step=overall_step,
            max_steps=self.max_steps,
            last_loss=self.last_loss,
        )

        if "train_loss" in self.active_group_summary or "train_runtime" in self.active_group_summary:
            message = (
                f"[第 {self.active_group_index + 1}/{self.total_groups} 组]"
                f"[{step}/{group_max_steps}] "
                f"训练器已完成当前组统计汇总"
            )
            logger.info(message)
            self._write_log(message)

    def on_group_end(self, model_path: str) -> None:
        """记录单组训练的总结信息。"""
        completed_step = min(
            self.max_steps or (self.active_group_base_step + self.active_group_max_steps),
            self.active_group_base_step + self.active_group_max_steps,
        )
        self.current_step = completed_step

        summary = {
            "group_index": self.active_group_index + 1,
            "target": self.active_group_target,
            "step": self.active_group_max_steps,
            "max_steps": self.active_group_max_steps,
            "last_loss": self.last_loss,
            "train_loss": self.active_group_summary.get("train_loss"),
            "train_runtime": self.active_group_summary.get("train_runtime"),
            "train_steps_per_second": self.active_group_summary.get("train_steps_per_second"),
            "train_samples_per_second": self.active_group_summary.get("train_samples_per_second"),
            "model_path": model_path,
        }
        self.group_summaries.append(summary)

        parts = [
            f"第 {summary['group_index']}/{self.total_groups} 组训练完成",
            f"target={summary['target']}",
            f"步数=[{summary['step']}/{summary['max_steps']}]",
        ]
        if summary["train_loss"] is not None:
            parts.append(f"平均损失={summary['train_loss']:.6f}")
        if summary["last_loss"] is not None:
            parts.append(f"最新损失={summary['last_loss']:.6f}")
        if summary["train_runtime"] is not None:
            parts.append(f"耗时={summary['train_runtime']:.2f}s")
        parts.append(f"模型路径={summary['model_path']}")

        message = "，".join(parts)
        logger.info(message)
        self._write_log(message)

    def on_training_end(self) -> None:
        """在整个任务训练结束时调用，输出任务总结。"""
        group_count = len(self.group_summaries)
        message = (
            f"训练任务结束: 任务 {self.job_id}, "
            f"完成组数 {group_count}/{self.total_groups}, "
            f"总步数 [{self.current_step}/{self.max_steps}]"
        )
        if self.last_loss is not None:
            message += f", 最新损失 {self.last_loss:.6f}"

        logger.info(message)
        self._write_log(message)

        for summary in self.group_summaries:
            parts = [
                f"总结 - 第 {summary['group_index']}/{self.total_groups} 组",
                f"target={summary['target']}",
            ]
            if summary["train_loss"] is not None:
                parts.append(f"平均损失={summary['train_loss']:.6f}")
            if summary["train_runtime"] is not None:
                parts.append(f"耗时={summary['train_runtime']:.2f}s")
            parts.append(f"模型路径={summary['model_path']}")
            self._write_log("，".join(parts))

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
        group_step = step - self.active_group_base_step
        if self.active_group_max_steps > 0 and group_step < 0:
            group_step = step

        if self.total_groups > 0 and self.active_group_max_steps > 0:
            parts = [
                f"[第 {self.active_group_index + 1}/{self.total_groups} 组]",
                f"[{group_step}/{self.active_group_max_steps}]",
            ]
        else:
            parts = [f"[步骤 {step}]"]

        if self.max_steps:
            parts.append(f"[总进度 {step}/{self.max_steps}]")

        if loss is not None:
            parts.append(f"损失={loss:.6f}")

        for key, value in kwargs.items():
            if value is not None:
                if isinstance(value, float):
                    if key == "learning_rate":
                        parts.append(f"{key}={value:.2e}")
                    else:
                        parts.append(f"{key}={value:.6f}")
                else:
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

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        """尽量将日志指标转换为 float。"""
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _format_loss(loss: Optional[float]) -> str:
        """统一格式化损失，避免输出 N/A。"""
        if loss is None:
            return "未上报"
        return f"{loss:.6f}"


class TrainerProgressCallback(TrainerCallback):
    """将 HuggingFace Trainer 日志转发到业务进度回调。"""

    def __init__(self, progress_callback: ProgressCallback) -> None:
        self.progress_callback = progress_callback

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self.progress_callback.check_cancel_requested()

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self.progress_callback.check_cancel_requested()

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self.progress_callback.on_trainer_log(
            step=state.global_step,
            max_steps=state.max_steps,
            logs=logs,
        )
