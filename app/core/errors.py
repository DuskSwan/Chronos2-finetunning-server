"""规范兼容接口的业务异常定义。"""

from dataclasses import dataclass


@dataclass
class ApiError(Exception):
    """带业务码和 HTTP 状态码的异常。"""

    code: int
    message: str
    http_status: int

