"""
ModelForge AI - Sliding Window Rate Limiting Engine
Implements token bucket and sliding window rate limiting using Redis / in-memory memory store.
"""

import time
from typing import Optional
from app.core.redis import get_redis_client
from app.core.exceptions import RateLimitExceededException
from app.core.config import settings
from app.core.logging import logger

_in_memory_rate_cache = {}


class RateLimiter:
    """Sliding-window rate limiter per API Key, User ID, or Client IP."""

    @staticmethod
    async def check_rate_limit(
        identifier: str,
        limit_per_minute: int = settings.RATE_LIMIT_PER_MINUTE,
    ) -> bool:
        """
        Check and record an access attempt in the sliding window.
        Raises RateLimitExceededException if quota exceeded.
        """
        now = time.time()
        window_start = now - 60.0
        key = f"rate_limit:{identifier}"

        try:
            client = await get_redis_client()
            # Redis sorted set for accurate sliding window
            pipeline = client.pipeline()
            pipeline.zremrangebyscore(key, 0, window_start)
            pipeline.zadd(key, {str(now): now})
            pipeline.zcard(key)
            pipeline.expire(key, 65)
            results = await pipeline.execute()

            current_requests = results[2]
            if current_requests > limit_per_minute:
                raise RateLimitExceededException(
                    message=f"Rate limit exceeded: maximum {limit_per_minute} requests per minute allowed."
                )
            return True
        except RateLimitExceededException:
            raise
        except Exception as e:
            # Fallback to local memory limiter
            records = _in_memory_rate_cache.get(identifier, [])
            records = [ts for ts in records if ts > window_start]
            if len(records) >= limit_per_minute:
                raise RateLimitExceededException(
                    message=f"Rate limit exceeded: maximum {limit_per_minute} requests per minute allowed."
                )
            records.append(now)
            _in_memory_rate_cache[identifier] = records
            return True


SlidingWindowRateLimiter = RateLimiter
