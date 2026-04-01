"""
Database initialization and schema management.
"""

from sqlalchemy.orm import Session

from app.db.models import Base
from app.db.session import engine


def init_db() -> None:
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


def get_db_engine():
    """Get the database engine."""
    return engine
