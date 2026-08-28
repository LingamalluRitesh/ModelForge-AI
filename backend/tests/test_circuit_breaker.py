"""
ModelForge AI - Circuit Breaker & Rate Limiter Unit Tests
"""

import pytest
import asyncio
from app.core.circuit_breaker import CircuitBreaker, CircuitState, CircuitBreakerOpenException
from app.core.rate_limiter import RateLimiter, RateLimitExceededException


@pytest.mark.asyncio
async def test_circuit_breaker_transitions():
    cb = CircuitBreaker(name="test_cb", failure_threshold=2, recovery_timeout=0.1, half_open_success_threshold=2)
    assert cb.state == CircuitState.CLOSED

    async def fail_func():
        raise ValueError("Service unavailable")

    async def success_func():
        return "ok"

    # 1. First failure
    with pytest.raises(ValueError):
        await cb.call(fail_func)
    assert cb.state == CircuitState.CLOSED

    # 2. Second failure -> Opens circuit
    with pytest.raises(ValueError):
        await cb.call(fail_func)
    assert cb.state == CircuitState.OPEN

    # 3. Call blocked immediately while OPEN
    with pytest.raises(CircuitBreakerOpenException):
        await cb.call(success_func)

    # 4. Wait for recovery timeout -> Transitions to HALF_OPEN on first call
    await asyncio.sleep(0.15)
    result_1 = await cb.call(success_func)
    assert result_1 == "ok"
    assert cb.state == CircuitState.HALF_OPEN

    # 5. Second successful call fulfills half_open_success_threshold -> Transitions to CLOSED
    result_2 = await cb.call(success_func)
    assert result_2 == "ok"
    assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_rate_limiter_in_memory():
    # Allow 2 requests per 60 seconds
    await RateLimiter.check_rate_limit(identifier="test_user_unique_123", limit_per_minute=2)
    await RateLimiter.check_rate_limit(identifier="test_user_unique_123", limit_per_minute=2)

    # 3rd request should raise RateLimitExceededException
    with pytest.raises(RateLimitExceededException):
        await RateLimiter.check_rate_limit(identifier="test_user_unique_123", limit_per_minute=2)
