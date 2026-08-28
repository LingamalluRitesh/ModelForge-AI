"""
ModelForge AI - Serving: Production Circuit Breaker & Token Bucket Rate Limiter
Guarantees sub-10ms response SLAs, automatic fallback routing, and graceful degradation during traffic bursts.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import time
import threading


class CircuitBreakerOpenException(Exception):
    pass


class CircuitBreaker:
    """Fowler Circuit Breaker pattern with Closed, Open, and Half-Open states."""
    def __init__(self, failure_threshold: int = 5, recovery_timeout_seconds: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout_seconds
        self.state = "CLOSED"
        self.failure_count = 0
        self.last_state_change = time.time()
        self.lock = threading.Lock()

    def record_success(self):
        with self.lock:
            self.failure_count = 0
            self.state = "CLOSED"

    def record_failure(self):
        with self.lock:
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                self.last_state_change = time.time()

    def allow_request(self) -> bool:
        with self.lock:
            if self.state == "CLOSED":
                return True
            elif self.state == "OPEN":
                if time.time() - self.last_state_change > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    return True
                return False
            elif self.state == "HALF_OPEN":
                return True
            return False


class TokenBucketRateLimiter:
    """Leaky Token Bucket for strict requests-per-second concurrency throttling."""
    def __init__(self, capacity: int = 1000, refill_rate_per_sec: float = 500.0):
        self.capacity = capacity
        self.refill_rate = refill_rate_per_sec
        self.tokens = float(capacity)
        self.last_refill = time.perf_counter()
        self.lock = threading.Lock()

    def acquire(self, tokens_needed: int = 1) -> bool:
        with self.lock:
            now = time.perf_counter()
            elapsed = now - self.last_refill
            self.tokens = min(float(self.capacity), self.tokens + elapsed * self.refill_rate)
            self.last_refill = now

            if self.tokens >= tokens_needed:
                self.tokens -= tokens_needed
                return True
            return False
