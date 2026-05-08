"""日志输出工具。"""

import json
from typing import Any


def to_pretty_log(value: Any) -> str:
    """将响应对象格式化为更易读的日志字符串。"""
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except TypeError:
        return str(value)
