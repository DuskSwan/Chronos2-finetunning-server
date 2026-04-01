"""数据集加载和转换服务。

支持从 CSV 和 Parquet 文件加载时间序列数据，并转换为 Chronos-2 训练所需的格式。
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


logger = logging.getLogger(__name__)


def load_data(
    file_path: str,
    target_column: str = "target",
    item_id_column: str = "item_id",
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """从文件加载数据。

    支持 CSV 和 Parquet 格式。

    Args:
        file_path: 数据文件路径。
        target_column: 目标值列名。
        item_id_column: 项目 ID 列名。
        timestamp_column: 时间戳列名。

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
    
    logger.info(f"数据加载成功: {file_path}，形状: {df.shape}")
    return df


def validate_required_columns(
    df: pd.DataFrame,
    target_column: str = "target",
    item_id_column: str = "item_id",
    timestamp_column: str = "timestamp",
) -> None:
    """验证 DataFrame 是否包含必需的列。

    Args:
        df: 数据框。
        target_column: 目标值列名。
        item_id_column: 项目 ID 列名。
        timestamp_column: 时间戳列名。

    Raises:
        ValueError: 如果缺少必需列。
    """
    required_columns = {target_column, item_id_column, timestamp_column}
    missing_columns = required_columns - set(df.columns)
    
    if missing_columns:
        raise ValueError(
            f"缺少必需列: {missing_columns}。"
            f"可用列: {list(df.columns)}"
        )


def convert_to_chronos_format(
    df: pd.DataFrame,
    target_column: str = "target",
    item_id_column: str = "item_id",
    timestamp_column: str = "timestamp",
) -> Dict[str, List[Any]]:
    """转换为 Chronos-2 fit() 所需的输入格式。

    返回格式：
    {
        "target": [时间序列1, 时间序列2, ...],
        "past_time_feat": [...],  # 可选
        "past_observed_indicator": [...],  # 可选
    }

    其中每个时间序列是一个列表。

    Args:
        df: 加载的数据框。
        target_column: 目标值列名。
        item_id_column: 项目 ID 列名。
        timestamp_column: 时间戳列名。

    Returns:
        Chronos-2 fit() 所需的字典格式。
    """
    validate_required_columns(df, target_column, item_id_column, timestamp_column)
    
    # 按 item_id 分组，按 timestamp 排序
    grouped = df.groupby(item_id_column)
    
    # 收集所有时序数据
    target_series_list = []
    
    for item_id, group in grouped:
        # 按时间戳排序
        group = group.sort_values(by=timestamp_column)
        
        # 提取目标值作为列表
        target_values = group[target_column].tolist()
        
        if target_values:  # 只添加非空序列
            target_series_list.append(target_values)
        
        logger.debug(
            f"项目 {item_id}: 加载 {len(target_values)} 个数据点"
        )
    
    if not target_series_list:
        raise ValueError("没有有效的时间序列数据")
    
    # 构造 Chronos 格式的输入字典
    chronos_input = {
        "target": target_series_list,
    }
    
    logger.info(f"转换为 Chronos-2 格式: {len(target_series_list)} 个时序")
    
    return chronos_input


def prepare_training_data(
    train_data_path: str,
    target_column: str = "target",
    item_id_column: str = "item_id",
    timestamp_column: str = "timestamp",
) -> Dict[str, List[Any]]:
    """准备训练数据。

    一体化函数：加载 → 验证 → 转换。

    Args:
        train_data_path: 训练数据文件路径。
        target_column: 目标值列名。
        item_id_column: 项目 ID 列名。
        timestamp_column: 时间戳列名。

    Returns:
        Chronos-2 fit() 所需的字典格式。
    """
    df = load_data(
        train_data_path,
        target_column=target_column,
        item_id_column=item_id_column,
        timestamp_column=timestamp_column,
    )
    
    chronos_input = convert_to_chronos_format(
        df,
        target_column=target_column,
        item_id_column=item_id_column,
        timestamp_column=timestamp_column,
    )
    
    return chronos_input


def prepare_validation_data(
    val_data_path: Optional[str],
    target_column: str = "target",
    item_id_column: str = "item_id",
    timestamp_column: str = "timestamp",
) -> Optional[Dict[str, List[Any]]]:
    """准备验证数据。

    Args:
        val_data_path: 验证数据文件路径，可为 None。
        target_column: 目标值列名。
        item_id_column: 项目 ID 列名。
        timestamp_column: 时间戳列名。

    Returns:
        Chronos-2 fit() 所需的验证数据字典，若 val_data_path 为 None 则返回 None。
    """
    if not val_data_path:
        logger.info("未提供验证数据")
        return None
    
    return prepare_training_data(
        val_data_path,
        target_column=target_column,
        item_id_column=item_id_column,
        timestamp_column=timestamp_column,
    )
