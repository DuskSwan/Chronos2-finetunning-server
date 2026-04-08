"""真实训练服务。

使用 Chronos-2 官方模型和微调接口执行真实的微调训练。
"""


import json
import inspect
from pathlib import Path
from typing import Any, Dict, Optional, Literal, TypedDict

from sqlalchemy.orm import Session

from app.callbacks.progress_callback import ProgressCallback, CancelledError
from app.services.dataset_service import prepare_input_data
from app.services.model_service import load_local_model

from chronos import BaseChronosPipeline, Chronos2Pipeline

from loguru import logger


class SelectedGroup(TypedDict):
    """目标列与协变量列的分组定义。"""

    target: str
    covariates: list[str]


def train_chronos2(
    db: Session,
    job_id: str,
    train_data_path: str,
    val_data_path: Optional[str],
    output_dir: str,
    log_path: str,
    prediction_length: int = 96,
    context_length: int = 512,
    finetune_mode: Literal['full', 'lora'] = "lora",
    learning_rate: float = 1e-4,
    num_steps: int = 1000,
    batch_size: int = 32,
    logging_steps: int = 100,
    finetuned_ckpt_name: str = "finetuned-ckpt",
    device: str = "cpu",
    selected_groups: Optional[list[SelectedGroup]] = None,
    **kwargs: Any
) -> str:
    """使用 Chronos-2 微调训练模型。

    此函数加载数据、准备训练环境、调用官方 Chronos-2 fit()、
    并通过 callback 更新进度。

    Args:
        db: 数据库会话。
        job_id: 任务 ID。
        train_data_path: 训练数据文件路径。
        val_data_path: 验证数据文件路径（可选）。
        output_dir: 输出目录。
        log_path: 日志文件路径。
        prediction_length: 预测长度。
        context_length: 上下文长度（默认 512）。
        finetune_mode: 微调模式（"lora" 或 "full"，默认 "lora"）。
        learning_rate: 学习率（默认 1e-4）。
        num_steps: 训练步数（默认 1000）。
        batch_size: 批大小（默认 32）。
        logging_steps: 日志间隔（默认 100）。
        finetuned_ckpt_name: 微调检查点名称。
        device: 设备（"cpu" 或 "cuda"，默认 "cpu"）。
        selected_groups: 目标列与协变量列的分组列表。
        **kwargs: 其他参数。

    Returns:
        已保存模型的路径。

    Raises:
        FileNotFoundError: 如果数据文件不存在。
        RuntimeError: 如果训练失败。
    """
    # 记录开始
    logger.info(
        f"开始 Chronos-2 微调训练"
        f" (job={job_id}, finetune_mode={finetune_mode})"
    )

    # 准备输出目录
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # 初始化回调
    callback = ProgressCallback(
        db_session=db,
        job_id=job_id,
        log_path=log_path,
        max_steps=num_steps,
    )
    callback.on_training_start()
    callback.check_cancel_requested()

    try:
        # 1. 加载和转换数据
        logger.info(f"加载训练数据: {train_data_path}")
        callback._write_log(f"加载训练数据: {train_data_path}")
        callback.check_cancel_requested()

        train_inputs = prepare_input_data(
            train_data_path,
            selected_groups=selected_groups,
        )
        logger.info(f"训练数据准备完成: shape={train_inputs.shape}")

        # 2. 准备验证数据
        validation_inputs = None
        if val_data_path:
            logger.info(f"加载验证数据: {val_data_path}")
            callback._write_log(f"加载验证数据: {val_data_path}")
            callback.check_cancel_requested()
            validation_inputs = prepare_input_data(
                val_data_path,
                selected_groups=selected_groups,
            )
            if validation_inputs is not None:
                logger.info(f"验证数据准备完成: shape={validation_inputs.shape}")

        # 3. 确定设备
        if "cuda" in device.lower():
            try:
                import torch
                if torch.cuda.is_available():
                    use_device = "cuda"
                    logger.info("使用 CUDA 设备进行训练")
                    callback._write_log("使用 CUDA 设备进行训练")
                else:
                    use_device = "cpu"
                    logger.warning("CUDA 不可用，回退到 CPU")
                    callback._write_log("警告: CUDA 不可用，使用 CPU")
            except ImportError:
                use_device = "cpu"
                logger.warning("PyTorch 未安装或 CUDA 不可用，使用 CPU")
        else:
            use_device = "cpu"
            logger.info("使用 CPU 进行训练")
            callback._write_log("使用 CPU 进行训练")

        # 4. 加载 Chronos-2 模型
        callback._write_log(f"加载base模型")
        callback.check_cancel_requested()

        # 使用上下文管理器自动处理设备
        pipeline: Chronos2Pipeline = load_local_model(device=use_device)

        logger.info(f"base模型加载成功")
        callback._write_log(f"模型加载成功")

        # 5. 调用 fit() 方法进行微调
        logger.info(
            f"开始微调: finetune_mode={finetune_mode}, "
            f"learning_rate={learning_rate}, "
            f"num_steps={num_steps}"
        )
        callback._write_log(
            f"开始微调: finetune_mode={finetune_mode}, "
            f"num_steps={num_steps}"
        )
        callback.check_cancel_requested()

        # 创建一个轻量的回调适配器（Chronos-2 的回调接口可能不同）
        class ChronosCallbackAdapter:
            """适配 Chronos-2 的回调接口。"""

            def __init__(self, progress_callback: ProgressCallback) -> None:
                self.progress_callback = progress_callback

            def on_step_end(self, step: int, loss: Optional[float] = None) -> None:
                """每步结束时调用。"""
                self.progress_callback.on_step_end(step, loss=loss)

        callback_adapter = ChronosCallbackAdapter(callback)

        # 调用微调（兼容不同版本的 Chronos-2 fit() 签名）
        fit_kwargs: Dict[str, Any] = dict(
            inputs=train_inputs,
            validation_inputs=validation_inputs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_steps=num_steps,
            prediction_length=prediction_length,
            context_length=context_length,
            logging_steps=logging_steps,
            # Note: extra kwargs 会进入 TrainingArguments
        )

        supports_finetune_mode = "finetune_mode" in inspect.signature(pipeline.fit).parameters
        if supports_finetune_mode:
            fit_kwargs["finetune_mode"] = finetune_mode
        else:
            if finetune_mode != "full":
                msg = (
                    "当前 Chronos 版本的 fit() 不支持 finetune_mode，"
                    "将忽略该参数并使用全量微调。"
                )
                logger.warning(msg)
                callback._write_log(f"警告: {msg}")

        fine_tuned_pipeline = pipeline.fit(**fit_kwargs)

        # 6. 保存微调后的模型
        model_save_path = output_dir_path / finetuned_ckpt_name
        model_save_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"保存微调后的模型到: {model_save_path}")
        callback._write_log(f"保存模型到: {model_save_path}")

        fine_tuned_pipeline.save_pretrained(str(model_save_path))

        logger.info(f"模型保存成功")
        callback._write_log(f"模型保存成功")

        # 7. 记录训练完成
        callback.on_training_end()

        return str(model_save_path)

    except CancelledError as e:
        logger.info(f"训练被取消: {e}")
        callback._write_log(str(e))
        raise
    except Exception as e:
        logger.error(f"训练失败: {e}", exc_info=True)
        callback.on_exception(e)
        raise


def mock_train(
    config: Dict[str, Any],
    steps: int = 5,
) -> str:
    """模拟微调过程（用于测试）。

    此函数模拟一个假的微调训练，不实际对数据进行处理或模型训练。
    本函数仅供测试使用，真实训练请使用 train_chronos2()。

    Args:
        config: 训练配置字典，包含 output_dir 等信息。
        steps: 训练步数。默认 5。

    Returns:
        模型检查点的路径（假的）。

    例：
        model_path = mock_train({"output_dir": "/path/to/artifacts/job-id"})
    """
    import random
    import time

    output_dir = Path(config.get("output_dir", "./artifacts"))

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    # 模拟训练过程
    print(f"[模拟训练器] 开始训练，总步数: {steps}")

    for step in range(1, steps + 1):
        # 模拟每步处理时间 0.2 ~ 0.5 秒
        sleep_time = random.uniform(0.2, 0.5)
        time.sleep(sleep_time)

        # 模拟损失下降
        loss = 1.0 - (step / steps) * 0.7  # 从 1.0 下降到 ~0.3
        loss = loss + random.uniform(-0.02, 0.02)  # 加入一点随机波动
        loss = max(loss, 0.1)  # 防止负值

        print(f"[模拟训练器] 第 {step}/{steps} 步，损失: {loss:.4f}")

    # 返回假的模型路径
    model_path = str(output_dir / "finetuned-ckpt")
    print(f"[模拟训练器] 训练完成，模型路径: {model_path}")

    return model_path
