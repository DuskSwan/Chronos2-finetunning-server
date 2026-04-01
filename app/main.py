"""
FastAPI 应用工厂和配置。
"""


from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import health, finetune
from app.core.config import get_settings
from app.core.paths import ensure_dir
from app.db.init_db import init_db
from app.services.queue_service import initialize_queue
from app.workers.trainer_worker import initialize_worker


from loguru import logger


def initialize_directories() -> None:
    """初始化应用目录。"""
    settings = get_settings()
    ensure_dir(settings.artifacts_root_resolved)
    ensure_dir(settings.logs_root_resolved)
    ensure_dir(settings.sqlite_db_path_resolved.parent)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。
    
    启动时初始化队列和 worker，关闭时清理资源。
    """
    # 启动事件
    logger.info("应用启动中...")
    settings = get_settings()
    initialize_queue()
    initialize_worker(settings)
    logger.info("队列和后台 Worker 已初始化")
    
    yield
    
    # 关闭事件
    logger.info("应用关闭中...")
    logger.info("清理完成")


def create_app() -> FastAPI:
    """
    创建并配置 FastAPI 应用。
    
    返回：
        配置好的 FastAPI 应用实例
    """
    # 初始化目录
    initialize_directories()
    
    # 初始化数据库
    init_db()
    
    # 创建应用，带生命周期管理
    app = FastAPI(
        title="Chronos-2 微调服务",
        version="0.2.0",
        description="用于微调 Chronos-2 时间序列模型的 API（支持异步训练）",
        lifespan=lifespan,
    )
    
    # 添加 CORS 中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 包含路由器
    app.include_router(health.router)
    app.include_router(finetune.router)
    
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
    )
