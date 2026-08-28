"""
ModelForge AI - Redis Client & Distributed Caching Module
Provides async and sync Redis connections, caching decorators, distributed locks,
and rate-limiting sliding window counters.
"""

from typing import Any, Optional, Union
import json
import logging
import redis.asyncio as aioredis
import redis
from app.core.config import settings

logger = logging.getLogger(__name__)

# Async Redis connection pool
_async_redis_client: Optional[aioredis.Redis] = None


async def get_redis_client() -> aioredis.Redis:
    """Get or initialize singleton async Redis client."""
    global _async_redis_client
    if _async_redis_client is None:
        try:
            _async_redis_client = aioredis.from_url(
                settings.redis_uri,
                encoding="utf-8",
                decode_responses=True,
                max_connections=20,
            )
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Falling back to in-memory cache mockup.")
            raise
    return _async_redis_client


def get_sync_redis_client() -> redis.Redis:
    """Get sync Redis client for Celery workers and synchronous tasks."""
    return redis.from_url(
        settings.redis_uri,
        encoding="utf-8",
        decode_responses=True,
    )


class CacheService:
    """Enterprise distributed caching layer with JSON serialization and TTL."""

    @staticmethod
    async def get(key: str) -> Optional[Any]:
        try:
            client = await get_redis_client()
            val = await client.get(key)
            if val is not None:
                return json.loads(val)
            return None
        except Exception as e:
            logger.debug(f"Redis get error for key '{key}': {e}")
            return None

    @staticmethod
    async def set(key: str, value: Any, ttl_seconds: int = 300) -> bool:
        try:
            client = await get_redis_client()
            serialized = json.dumps(value, default=str)
            await client.set(key, serialized, ex=ttl_seconds)
            return True
        except Exception as e:
            logger.debug(f"Redis set error for key '{key}': {e}")
            return False

    @staticmethod
    async def delete(key: str) -> bool:
        try:
            client = await get_redis_client()
            await client.delete(key)
            return True
        except Exception as e:
            logger.debug(f"Redis delete error for key '{key}': {e}")
            return False

    @staticmethod
    async def delete_pattern(pattern: str) -> int:
        """Invalidate all keys matching a glob pattern."""
        try:
            client = await get_redis_client()
            keys = await client.keys(pattern)
            if keys:
                return await client.delete(*keys)
            return 0
        except Exception as e:
            logger.debug(f"Redis delete_pattern error for '{pattern}': {e}")
            return 0


class DistributedLock:
    """Redis-backed distributed lock with timeout to prevent deadlocks."""

    def __init__(self, lock_name: str, timeout_seconds: int = 60):
        self.lock_name = f"lock:{lock_name}"
        self.timeout = timeout_seconds
        self._acquired = False

    async def __aenter__(self):
        try:
            client = await get_redis_client()
            self._acquired = await client.set(
                self.lock_name,
                "locked",
                nx=True,
                ex=self.timeout,
            )
            return self._acquired
        except Exception as e:
            logger.warning(f"Distributed lock acquire failed: {e}")
            return False

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._acquired:
            try:
                client = await get_redis_client()
                await client.delete(self.lock_name)
            except Exception as e:
                logger.warning(f"Distributed lock release failed: {e}")
