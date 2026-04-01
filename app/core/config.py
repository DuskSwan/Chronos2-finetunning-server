"""
Configuration management for the application.
"""

from pathlib import Path
from typing import Optional

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")
    
    # Server
    host: str = "127.0.0.1"
    port: int = 8000
    
    # Database
    sqlite_db_path: str = "./data/finetune.db"
    
    # Paths
    artifacts_root: str = "./artifacts"
    logs_root: str = "./logs"
    
    # Model
    default_model_id: str = "amazon/chronos-2"
    
    @property
    def sqlite_db_path_resolved(self) -> Path:
        """Get resolved SQLite DB path."""
        return Path(self.sqlite_db_path).resolve()
    
    @property
    def artifacts_root_resolved(self) -> Path:
        """Get resolved artifacts root path."""
        return Path(self.artifacts_root).resolve()
    
    @property
    def logs_root_resolved(self) -> Path:
        """Get resolved logs root path."""
        return Path(self.logs_root).resolve()


def get_settings() -> Settings:
    """Get application settings instance."""
    return Settings()
