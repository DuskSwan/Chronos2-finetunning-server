"""数据集加载和转换服务。

支持从 CSV 和 Parquet 文件加载时间序列数据，并转换为 Chronos-2 训练所需的格式。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
from loguru import logger


def load_data(
    file_path: str,
    target_columns: List[str] | None = None,
) -> pd.DataFrame:
    """从文件加载数据。

    支持 CSV 和 Parquet 格式。

    Args:
        file_path: 数据文件路径。
        target_columns: 目标值列名列表。如果提供，将验证这些列是否存在。不提供则使用全部的列。

    Returns:
        加载后的 DataFrame。

    Raises:
        FileNotFoundError: 如果文件不存在。
        ValueError: 如果文件格式不支持。
    """
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        raise FileNotFoundError(f"数据文件不存在: {file_path}")
    
    if file_path_obj.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
    elif file_path_obj.suffix.lower() == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(
            f"不支持的文件格式: {file_path_obj.suffix}。"
            f"支持的格式: .csv, .parquet"
        )
    if target_columns:
        missing_columns = set(target_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"缺少目标列: {missing_columns}。"
                f"可用列: {list(df.columns)}"
            )
        df = df[target_columns]
    
    logger.info(f"数据加载成功: {file_path}，形状: {df.shape}")
    return df


def prepare_input_data(
    train_data_path: str,
    target_columns: List[str] | None = None,
) -> np.ndarray:
    """准备训练数据。

    一体化函数：加载 → 验证 → 转换。

    Args:
        train_data_path: 训练数据文件路径。
        target_columns: 目标值列名列表。

    Returns:
        Chronos-2 fit() 所需的 (batch_size, num_variates, history_length) 格式的 numpy 数组。
    """
    df = load_data(
        train_data_path,
        target_columns=target_columns,
    )
    
    # 转为可写的数值数组，避免非数值/只读数组导致训练失败
    array = df.to_numpy(copy=True).astype("float32", copy=False)
    array = array.transpose()  # 转置为 (num_variates, history_length)
    array = array[np.newaxis, :] # 增加 batch_size 维度，变为 (1, num_variates, history_length)
    
    return array
