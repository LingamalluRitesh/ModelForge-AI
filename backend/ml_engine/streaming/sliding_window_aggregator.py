"""
ModelForge AI - ML Engine: Streaming Aggregation & Probabilistic Data Structures
Implements Count-Min Sketch for frequency estimation, HyperLogLog for cardinality estimation,
and Exponential Moving Average (EMA) for real-time latency & drift streaming features.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math
import hashlib
import numpy as np


class CountMinSketch:
    """Cormode & Muthukrishnan Count-Min Sketch for streaming frequency estimation."""

    def __init__(self, width: int = 1000, depth: int = 5):
        self.width = width
        self.depth = depth
        self.table = np.zeros((depth, width), dtype=np.int64)

    def _hash(self, item: str, seed: int) -> int:
        h = hashlib.sha256(f"{seed}:{item}".encode("utf-8")).hexdigest()
        return int(h, 16) % self.width

    def add(self, item: str, count: int = 1):
        for d in range(self.depth):
            w = self._hash(item, d)
            self.table[d, w] += count

    def estimate(self, item: str) -> int:
        estimates = [self.table[d, self._hash(item, d)] for d in range(self.depth)]
        return int(min(estimates))


class HyperLogLog:
    """Flajolet et al. HyperLogLog cardinality estimator using leading zeros."""

    def __init__(self, p: int = 10):
        if not 4 <= p <= 16:
            raise ValueError("Precision p must be in [4, 16]")
        self.p = p
        self.m = 1 << p  # 2^p registers
        self.registers = np.zeros(self.m, dtype=np.uint8)

        # Alpha correction constant
        if self.m == 16:
            self.alpha = 0.673
        elif self.m == 32:
            self.alpha = 0.697
        elif self.m == 64:
            self.alpha = 0.709
        else:
            self.alpha = 0.7213 / (1.0 + 1.079 / self.m)

    def _clz(self, x: int, max_bits: int = 64) -> int:
        """Count leading zeros."""
        if x == 0:
            return max_bits
        binary_str = bin(x)[2:].zfill(max_bits)
        return len(binary_str) - len(binary_str.lstrip("0")) + 1

    def add(self, item: str):
        h = int(hashlib.sha256(item.encode("utf-8")).hexdigest()[:16], 16)
        # Register index: first p bits
        j = h >> (64 - self.p)
        # Trailing hash bits
        w = h & ((1 << (64 - self.p)) - 1)
        rank = self._clz(w, max_bits=64 - self.p)

        self.registers[j] = max(self.registers[j], rank)

    def cardinality(self) -> int:
        # Harmonic mean of 2^-M[j]
        indicator = float(np.sum(2.0 ** (-self.registers.astype(float))))
        E = self.alpha * (self.m ** 2) / indicator

        # Small range correction
        if E <= 2.5 * self.m:
            V = np.sum(self.registers == 0)
            if V > 0:
                E = self.m * math.log(self.m / V)

        return int(E)


class ExponentialMovingAverage:
    """Online Real-Time Exponential Moving Average (EMA) and Variance estimator."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.mean: Optional[float] = None
        self.variance: float = 0.0

    def update(self, value: float):
        if self.mean is None:
            self.mean = value
            self.variance = 0.0
        else:
            delta = value - self.mean
            self.mean += self.alpha * delta
            # Incremental variance update
            self.variance = (1.0 - self.alpha) * (self.variance + self.alpha * (delta ** 2))

    @property
    def std(self) -> float:
        return math.sqrt(max(0.0, self.variance))
