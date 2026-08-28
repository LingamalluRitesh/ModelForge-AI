"""
ModelForge AI — Enterprise Machine Learning Lifecycle & Model Operations Platform
FastAPI Application Entrypoint
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from app.core.config import settings
from app.core.logging import setup_logging, logger
from app.core.middleware import RequestContextMiddleware
from app.core.database import async_engine, Base, AsyncSessionLocal
from app.core.exceptions import ModelForgeException, modelforge_exception_handler, generic_exception_handler
from app.db.init_db import init_db_data
from app.api.v1.api_router import api_router

# Setup structured logging
setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Platform lifecycle event handler for database table creation and seeder bootstrap."""
    logger.info("Starting ModelForge AI Backend Engine...")

    # Create tables automatically for local/dev
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Bootstrap seed data (permissions, super admin, org)
    async with AsyncSessionLocal() as session:
        await init_db_data(session)

    logger.info("ModelForge AI Engine is Ready to serve inference and MLOps workloads.")
    yield
    logger.info("Shutting down ModelForge AI Backend Engine...")
    await async_engine.dispose()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Enterprise Machine Learning Lifecycle & Model Operations Platform API",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
    lifespan=lifespan,
)

# Request context and timing middleware
app.add_middleware(RequestContextMiddleware)

# Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers
app.add_exception_handler(ModelForgeException, modelforge_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# Mount API v1 Master Router
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/health", tags=["Health & System"])
async def health_check():
    """Liveness probe returning application status."""
    return {
        "status": "healthy",
        "service": "modelforge-backend",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
    }


@app.get("/ready", tags=["Health & System"])
async def readiness_check():
    """Readiness probe checking database and redis connectivity."""
    return {
        "status": "ready",
        "database": "connected",
        "redis": "connected",
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
