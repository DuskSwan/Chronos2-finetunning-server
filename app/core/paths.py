"""
应用的路径工具。
"""

from pathlib import Path


def get_project_root() -> Path:
    """获取项目根目录。"""
    return Path(__file__).parent.parent.parent


def ensure_dir(path: Path) -> Path:
    """确保目录存在，如果不存在则创建。"""
    path.mkdir(parents=True, exist_ok=True)
    return path
