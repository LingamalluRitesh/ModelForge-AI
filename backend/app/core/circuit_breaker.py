"""
ModelForge AI - Circuit Breaker Pattern Implementation
Protects real-time inference endpoints, third-party APIs, and downstream services
from cascading failures and latency degradation.
"""

import time
import asyncio
from enum import Enum
from typing import Callable, Any
from app.core.exceptions import CircuitBreakerOpenException
from app.core.logging import logger


class CircuitState(str, Enum):
    CLOSED = "CLOSED"      # Healthy, traffic passing through
    OPEN = "OPEN"          # Tripped, immediately failing fast
    HALF_OPEN = "HALF_OPEN"# Testing recovery with trial requests


CircuitBreakerState = CircuitState


class CircuitBreaker:
    """Enterprise circuit breaker with configurable failure threshold and recovery timeout."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_success_threshold: int = 2,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_success_threshold = half_open_success_threshold

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_state_change = time.time()

    def _update_state(self):
        """Transition from OPEN to HALF_OPEN when recovery timeout passes."""
        now = time.time()
        if self.state == CircuitState.OPEN and (now - self.last_state_change) > self.recovery_timeout:
            self.state = CircuitState.HALF_OPEN
            self.success_count = 0
            self.last_state_change = now
            logger.info(f"Circuit breaker '{self.name}' transitioned from OPEN to HALF_OPEN.")

    def record_success(self):
        """Record a successful request."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.half_open_success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                self.last_state_change = time.time()
                logger.info(f"Circuit breaker '{self.name}' recovered and is now CLOSED.")
        elif self.state == CircuitState.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)

    def record_failure(self):
        """Record a failed request."""
        self.failure_count += 1
        if self.state in (CircuitState.CLOSED, CircuitState.HALF_OPEN):
            if self.failure_count >= self.failure_threshold or self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self.last_state_change = time.time()
                logger.warning(
                    f"Circuit breaker '{self.name}' tripped to OPEN! "
                    f"Failures: {self.failure_count}, Threshold: {self.failure_threshold}"
                )

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function protected by this circuit breaker."""
        self._update_state()

        if self.state == CircuitState.OPEN:
            raise CircuitBreakerOpenException(service_name=self.name)

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            raise e
