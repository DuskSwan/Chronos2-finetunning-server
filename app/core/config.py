"""
应用配置管理。
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用设置。"""
    
    model_config: SettingsConfigDict = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )
    
    # 服务器
    host: str = "127.0.0.1"
    port: int = 8000
    
    # 数据库
    sqlite_db_path: str = "./data/finetune.db"
    
    # 路径
    artifacts_root: str = "./artifacts"
    logs_root: str = "./logs"
    release_path: str = "./release"

    # 模型
    raw_model_cache_dir: str = "./data/model_cache"
    device: str = "cuda"
    logging_steps: int = 100
    finetuned_ckpt_name: str = "finetuned-ckpt"

    # 开关
    save_request_artifacts: bool = True # 是否保存请求 JSON 到产物目录 
    api_bearer_token: str = ""
    chunk_infer_cache_ttl_seconds: int = 1800
    chunk_infer_cache_max_tasks: int = 128
    
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

    @property
    def release_path_resolved(self) -> Path:
        """获取解析后的发布目录路径。"""
        return Path(self.release_path).resolve()


def get_settings() -> Settings:
    """获取应用设置实例。"""
    return Settings()
