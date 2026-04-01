"""
数据库初始化和模式管理。
"""

from sqlalchemy.orm import Session

from app.db.models import Base
from app.db.session import engine


def init_db() -> None:
    """初始化数据库表。"""
    Base.metadata.create_all(bind=engine)


def get_db_engine():
    """获取数据库引擎。"""
    return engine
