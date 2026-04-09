"""数据集加载和转换服务。

支持从 CSV 和 Parquet 文件加载时间序列数据，并转换为 Chronos-2 训练所需的格式。
"""

from pathlib import Path
from typing import List, TypedDict

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


class SelectedGroup(TypedDict):
    """目标列与协变量列的分组定义。"""

    target: str
    covariates: List[str]


def prepare_input_data(
    train_data_path: str,
    selected_groups: List[SelectedGroup],
) -> list[dict]:
    """准备训练数据。

    一体化函数：加载 → 验证 → 转换。

    Args:
        train_data_path: 训练数据文件路径。
        selected_groups: 目标列与协变量列的分组列表，格式为
            { "target": "target_column", "covariates": ["col1","col2",...] }

    Returns:
        字典组成的列表，每一个字典是 Chronos-2 fit() 可接受的字典，形如
        {
            "target": np.ndarray(shape=(history_length,)),
            "past_covariates": {
                "temp": np.ndarray(shape=(history_length,)),
                "pressure": np.ndarray(shape=(history_length,)),
            },
        }
    """

    df = load_data(train_data_path)
    res = []

    for group in selected_groups:
        tar_col = group['target']
        cov_cols = group['covariates']
        res.append({})
        assert(tar_col in df.columns)
        # Ensure arrays are writable for PyTorch (avoid non-writable numpy warning)
        res[-1]['target'] = df[tar_col].to_numpy(copy=True)
        past_covariates = {}
        for col in cov_cols:
            assert(col in df.columns)
            past_covariates[col] = df[col].to_numpy(copy=True)
        if past_covariates:
            res[-1]['past_covariates'] = past_covariates
    
    return res
