"""
FastAPI 应用工厂和配置。
"""


from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import health, finetune, tools, train_jobs, model_publish
from app.core.config import get_settings
from app.core.errors import ApiError
from app.core.paths import ensure_dir
from app.db.init_db import init_db
from app.services.queue_service import initialize_queue
from app.services.train_job_adapter import normalize_api_error
from app.workers.trainer_worker import initialize_worker


from loguru import logger


def _is_spec_route(request: Request) -> bool:
    return request.url.path.startswith("/api/v1/train_jobs")

def _is_model_publish_route(request: Request) -> bool:
    return request.url.path == "/api/model/publish"


def register_spec_exception_handlers(app: FastAPI) -> None:
    """为规范兼容路由注册异常处理。"""

    @app.exception_handler(ApiError)
    async def handle_api_error(request: Request, exc: ApiError) -> JSONResponse:
        if not _is_spec_route(request):
            raise exc
        return JSONResponse(
            status_code=exc.http_status,
            content={"code": exc.code, "message": exc.message, "data": {}},
        )

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        if _is_model_publish_route(request):
            first = exc.errors()[0] if exc.errors() else {}
            msg = first.get("msg", "invalid parameter")
            return JSONResponse(
                status_code=200,
                content={"code": 500, "message": msg, "data": None},
            )
        if not _is_spec_route(request):
            return await request_validation_exception_handler(request, exc)
        err = normalize_api_error(exc)
        return JSONResponse(
            status_code=err.http_status,
            content={"code": err.code, "message": err.message, "data": {}},
        )


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

    register_spec_exception_handlers(app)
    
    # 包含路由器
    app.include_router(health.router)
    app.include_router(finetune.router)
    app.include_router(tools.router)
    app.include_router(train_jobs.router)
    app.include_router(model_publish.router)
    
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
