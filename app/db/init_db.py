"""
数据库初始化和模式管理。
"""

from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from app.db.models import Base
from app.db.session import engine


def _ensure_model_paths_column() -> None:
    """确保 finetune_jobs 表存在 model_paths 列（兼容旧库）。"""
    inspector = inspect(engine)
    if "finetune_jobs" not in inspector.get_table_names():
        return

    columns = {col["name"] for col in inspector.get_columns("finetune_jobs")}
    if "model_paths" in columns:
        return

    # SQLite 仅支持 ADD COLUMN，不支持 DROP/RENAME
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE finetune_jobs ADD COLUMN model_paths TEXT"))


def init_db() -> None:
    """初始化数据库表。"""
    Base.metadata.create_all(bind=engine)
    _ensure_model_paths_column()


def get_db_engine():
    """获取数据库引擎。"""
    return engine
