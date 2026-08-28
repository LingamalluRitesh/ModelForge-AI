"""
ModelForge AI - Database Core Engine
Async and Sync SQLAlchemy session factories, declarative base, and lifecycle handlers.
Supports PostgreSQL (asyncpg/psycopg2) and SQLite (aiosqlite/sqlite3) for testing.
"""

from typing import AsyncGenerator, Generator
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from app.core.config import settings

# Determine async connection string and engine settings
if "sqlite" in settings.async_database_uri:
    async_engine = create_async_engine(
        settings.async_database_uri,
        echo=settings.DB_ECHO,
        future=True,
    )
    sync_engine = create_engine(
        settings.sync_database_uri,
        echo=settings.DB_ECHO,
        future=True,
    )
else:
    async_engine = create_async_engine(
        settings.async_database_uri,
        echo=settings.DB_ECHO,
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
        pool_timeout=settings.DB_POOL_TIMEOUT,
        pool_pre_ping=True,
        future=True,
    )
    sync_engine = create_engine(
        settings.sync_database_uri,
        echo=settings.DB_ECHO,
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
        pool_timeout=settings.DB_POOL_TIMEOUT,
        pool_pre_ping=True,
        future=True,
    )

# Async and Sync session makers
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

SyncSessionLocal = sessionmaker(
    bind=sync_engine,
    class_=Session,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


class Base(DeclarativeBase):
    """Base declarative class for all SQLAlchemy ORM models."""
    pass


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for providing an async database session per request."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def get_sync_db() -> Generator[Session, None, None]:
    """Dependency for providing a sync database session (used in background workers)."""
    db = SyncSessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
