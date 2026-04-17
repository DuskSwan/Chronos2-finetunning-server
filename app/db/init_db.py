"""
数据库初始化和模式管理。
"""

from sqlalchemy import inspect, text

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


def _ensure_finetune_job_losses_table() -> None:
    """确保 finetune_job_losses 表存在。"""
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    if "finetune_job_losses" in tables:
        columns = {
            col["name"] for col in inspector.get_columns("finetune_job_losses")
        }
        # 旧版本没有 group_index，需要重建表结构（不兼容旧唯一键）
        if "group_index" not in columns:
            with engine.begin() as conn:
                conn.execute(
                    text(
                        "ALTER TABLE finetune_job_losses "
                        "RENAME TO finetune_job_losses_old"
                    )
                )
                conn.execute(
                    text(
                        """
                        CREATE TABLE IF NOT EXISTS finetune_job_losses (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            job_id VARCHAR(36) NOT NULL,
                            group_index INTEGER NOT NULL,
                            step INTEGER NOT NULL,
                            loss FLOAT NOT NULL,
                            created_at DATETIME NOT NULL,
                            CONSTRAINT uq_job_group_step UNIQUE (job_id, group_index, step),
                            FOREIGN KEY(job_id) REFERENCES finetune_jobs(id) ON DELETE CASCADE
                        )
                        """
                    )
                )
                conn.execute(
                    text(
                        """
                        INSERT INTO finetune_job_losses
                        (job_id, group_index, step, loss, created_at)
                        SELECT job_id, 0, step, loss, created_at
                        FROM finetune_job_losses_old
                        """
                    )
                )
                conn.execute(text("DROP TABLE finetune_job_losses_old"))
                conn.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS ix_finetune_job_losses_job_id "
                        "ON finetune_job_losses(job_id)"
                    )
                )
        return

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS finetune_job_losses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id VARCHAR(36) NOT NULL,
                    group_index INTEGER NOT NULL,
                    step INTEGER NOT NULL,
                    loss FLOAT NOT NULL,
                    created_at DATETIME NOT NULL,
                    CONSTRAINT uq_job_group_step UNIQUE (job_id, group_index, step),
                    FOREIGN KEY(job_id) REFERENCES finetune_jobs(id) ON DELETE CASCADE
                )
                """
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_finetune_job_losses_job_id "
                "ON finetune_job_losses(job_id)"
            )
        )


def init_db() -> None:
    """初始化数据库表。"""
    Base.metadata.create_all(bind=engine)
    _ensure_model_paths_column()
    _ensure_finetune_job_losses_table()


def get_db_engine():
    """获取数据库引擎。"""
    return engine
