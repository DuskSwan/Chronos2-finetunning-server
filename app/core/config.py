"""
应用配置管理。
"""

from pathlib import Path
from typing import Optional

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用设置。"""
    
    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")
    
    # 服务器
    host: str = "127.0.0.1"
    port: int = 8000
    
    # 数据库
    sqlite_db_path: str = "./data/finetune.db"
    
    # 路径
    artifacts_root: str = "./artifacts"
    logs_root: str = "./logs"
    
    # 模型
    default_model_id: str = "amazon/chronos-2"
    
    @property
    def sqlite_db_path_resolved(self) -> Path:
        """获取解析后的 SQLite 数据库路径。"""
        return Path(self.sqlite_db_path).resolve()
    
    @property
    def artifacts_root_resolved(self) -> Path:
        """获取解析后的产物根目录路径。"""
        return Path(self.artifacts_root).resolve()
    
    @property
    def logs_root_resolved(self) -> Path:
        """获取解析后的日志根目录路径。"""
        return Path(self.logs_root).resolve()


def get_settings() -> Settings:
    """获取应用设置实例。"""
    return Settings()
