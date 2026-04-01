"""回调模块。

用于在训练过程中捕获进度和性能指标，回写到数据库。
"""

from app.callbacks.progress_callback import ProgressCallback

__all__ = ["ProgressCallback"]
