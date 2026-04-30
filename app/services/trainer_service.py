"""真实训练服务。

使用 Chronos-2 官方模型和微调接口执行真实的微调训练。
"""


import json
import inspect
from pathlib import Path
from typing import Any, Dict, Optional, Literal, TypedDict

from sqlalchemy.orm import Session

from app.callbacks.progress_callback import (
    ProgressCallback,
    TrainerProgressCallback,
    CancelledError,
)
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
    # val_data_path: Optional[str],
    output_dir: str,
    log_path: str,
    selected_groups: list[SelectedGroup],
    prediction_length: int = 96,
    context_length: int = 512,
    finetune_mode: Literal['full', 'lora'] = "lora",
    learning_rate: float = 1e-4,
    num_steps: int = 1000,
    batch_size: int = 32,
    logging_steps: int = 100,
    finetuned_ckpt_name: str = "finetuned-ckpt",
    device: str = "cpu",
    **kwargs: Any
) -> dict[str, str]:
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
        target 到已保存模型路径的映射。

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
    total_steps = num_steps * len(selected_groups)
    callback = ProgressCallback(
        db_session=db,
        job_id=job_id,
        log_path=log_path,
        max_steps=total_steps,
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
        logger.info(f"训练数据准备完成: group_num={len(train_inputs)}")

        # 2. 准备验证数据
        # validation_inputs = None
        # if val_data_path:
        #     logger.info(f"加载验证数据: {val_data_path}")
        #     callback._write_log(f"加载验证数据: {val_data_path}")
        #     callback.check_cancel_requested()
        #     validation_inputs = prepare_input_data(
        #         val_data_path,
        #         selected_groups=selected_groups,
        #     )
        #     if validation_inputs is not None:
        #         logger.info(f"验证数据准备完成: shape={validation_inputs.shape}")

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

        target_model_map: dict[str, str] = {}
        for i, input_dict in enumerate(train_inputs):
            tar_col = selected_groups[i]["target"]
            callback.on_group_start(
                group_index=i,
                total_groups=len(train_inputs),
                target=tar_col,
                group_max_steps=num_steps,
            )

            # 调用微调
            fit_kwargs: Dict[str, Any] = dict(
                inputs=[input_dict],
                # validation_inputs=validation_inputs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                num_steps=num_steps,
                prediction_length=prediction_length,
                context_length=context_length,
                logging_steps=logging_steps,
                finetune_mode=finetune_mode,
                callbacks=[TrainerProgressCallback(callback)],
                disable_tqdm=True,
                remove_printer_callback=True,
                # Note: extra kwargs 会进入 TrainingArguments
            )
            assert "finetune_mode" in inspect.signature(pipeline.fit).parameters

            fine_tuned_pipeline = pipeline.fit(**fit_kwargs)

            # 6. 保存微调后的模型
            model_save_path = output_dir_path / f"{finetuned_ckpt_name}_{tar_col}"
            model_save_path.mkdir(parents=True, exist_ok=True)

            logger.info(f"第{i}相关组训练 - 保存微调后的模型到: {model_save_path}")
            callback._write_log(f"保存模型到: {model_save_path}")

            fine_tuned_pipeline.save_pretrained(str(model_save_path))

            logger.info(f"模型保存成功")
            callback._write_log(f"模型保存成功")

            # 7. 记录训练完成
            callback.on_group_end(model_path=str(model_save_path))
            target_model_map[tar_col] = str(model_save_path)

        callback.on_training_end()
        return target_model_map

    except CancelledError as e:
        logger.info(f"训练被取消: {e}")
        callback._write_log(str(e))
        raise
    except Exception as e:
        logger.error(f"训练失败: {e}", exc_info=True)
        callback.on_exception(e)
        raise
