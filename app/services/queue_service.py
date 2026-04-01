"""本地任务队列服务。

使用内存队列管理待处理的微调任务。设计上易于后续替换为
真正的消息队列（如 RabbitMQ、Redis）。
"""

from queue import Queue
from typing import Optional


class JobQueue:
    """单进程本地内存队列。"""

    def __init__(self, maxsize: int = 0) -> None:
        """初始化队列。

        Args:
            maxsize: 队列最大大小，0 表示无限制。
        """
        self._queue: Queue[str] = Queue(maxsize=maxsize)

    def enqueue(self, job_id: str) -> None:
        """将任务 ID 放入队列。

        Args:
            job_id: 任务唯一标识符。
        """
        self._queue.put(job_id, block=False)

    def dequeue(self, timeout: Optional[float] = None) -> Optional[str]:
        """从队列取出任务 ID。

        Args:
            timeout: 等待超时时间（秒）。None 表示永久等待。

        Returns:
            任务 ID，若超时返回 None。
        """
        try:
            return self._queue.get(timeout=timeout)
        except Exception:
            return None

    def size(self) -> int:
        """获取队列中的任务数量。

        Returns:
            队列大小。
        """
        return self._queue.qsize()

    def is_empty(self) -> bool:
        """检查队列是否为空。

        Returns:
            True 若队列为空，否则 False。
        """
        return self._queue.empty()


# 全局队列实例
_global_queue: Optional[JobQueue] = None


def get_job_queue() -> JobQueue:
    """获取全局工作队列实例。

    Returns:
        JobQueue 实例。
    """
    global _global_queue
    if _global_queue is None:
        _global_queue = JobQueue()
    return _global_queue


def initialize_queue() -> JobQueue:
    """初始化全局工作队列。

    Returns:
        初始化后的 JobQueue 实例。
    """
    global _global_queue
    _global_queue = JobQueue()
    return _global_queue
