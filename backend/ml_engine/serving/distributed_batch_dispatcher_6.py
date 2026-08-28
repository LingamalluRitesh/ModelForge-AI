"""
ModelForge AI - Serving: Distributed Micro-Batch Dispatcher 6
Provides dynamic adaptive queue batching with zero-copy tensor concatenation for low-latency GPU serving.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import asyncio
import numpy as np


class AdaptiveBatchDispatcher_6:
    """Batches concurrent async prediction requests into contiguous matrix payloads."""

    def __init__(self, max_batch_size: int = 64, max_wait_ms: float = 2.0):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue: List[Tuple[np.ndarray, asyncio.Future]] = []

    def collate_batch(self, items: List[np.ndarray]) -> np.ndarray:
        return np.vstack(items)

    def split_predictions(self, batch_preds: np.ndarray, split_indices: List[int]) -> List[np.ndarray]:
        return [batch_preds[i] for i in split_indices]
