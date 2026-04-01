"""
FastAPI application factory and configuration.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import health, finetune
from app.core.config import get_settings
from app.core.paths import ensure_dir
from app.db.init_db import init_db


def initialize_directories() -> None:
    """Initialize application directories."""
    settings = get_settings()
    ensure_dir(settings.artifacts_root_resolved)
    ensure_dir(settings.logs_root_resolved)
    ensure_dir(settings.sqlite_db_path_resolved.parent)


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Returns:
        Configured FastAPI app instance
    """
    # Initialize directories
    initialize_directories()
    
    # Initialize database
    init_db()
    
    # Create app
    app = FastAPI(
        title="Chronos-2 Fine-tuning Service",
        version="0.1.0",
        description="API for fine-tuning Chronos-2 time series models",
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
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
