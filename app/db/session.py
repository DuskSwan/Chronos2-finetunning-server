"""
数据库会话管理。
"""

from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.core.paths import ensure_dir


def get_db_url(db_path: Path) -> str:
    """获取 SQLite 数据库 URL。"""
    # 确保父目录存在
    ensure_dir(db_path.parent)
    return f"sqlite:///{db_path.as_posix()}"


def create_session_factory(db_url: str) -> sessionmaker:
    """创建 SQLAlchemy 会话工厂。"""
    engine = create_engine(
        db_url,
        connect_args={"check_same_thread": False},
        echo=False,
    )
    
    # 为 SQLite 启用外键
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
    
    return sessionmaker(autocommit=False, autoflush=False, bind=engine), engine


settings = get_settings()
db_url = get_db_url(settings.sqlite_db_path_resolved)
SessionLocal, engine = create_session_factory(db_url)


def get_db() -> Generator[Session, None, None]:
    """用于获取数据库会话的依赖。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
