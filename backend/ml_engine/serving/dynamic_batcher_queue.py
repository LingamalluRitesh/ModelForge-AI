"""
ModelForge AI - Serving: Adaptive Dynamic Micro-Batch Queue
Aggregates concurrent incoming real-time inference requests into tensor batches
balancing GPU throughput saturation against sub-10ms maximum latency timeout budgets.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import asyncio
import time


class BatchRequestItem:
    def __init__(self, request_id: str, input_features: np.ndarray, future: asyncio.Future):
        self.request_id = request_id
        self.input_features = input_features
        self.future = future
        self.arrival_time = time.perf_counter()


class DynamicBatchingEngine:
    """Async dynamic micro-batch scheduler."""
    def __init__(
        self,
        scoring_fn: Callable[[np.ndarray], np.ndarray],
        max_batch_size: int = 64,
        max_latency_timeout_ms: float = 8.0,
    ):
        self.scoring_fn = scoring_fn
        self.max_batch_size = max_batch_size
        self.max_timeout = max_latency_timeout_ms / 1000.0
        self.queue: List[BatchRequestItem] = []
        self.lock = asyncio.Lock()
        self.is_running = False

    async def enqueue(self, request_id: str, features: np.ndarray) -> np.ndarray:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        item = BatchRequestItem(request_id, features, future)

        async with self.lock:
            self.queue.append(item)

        return await future

    async def batch_loop(self):
        """Continuous micro-batch extraction loop."""
        self.is_running = True
        while self.is_running:
            await asyncio.sleep(0.001)

            async with self.lock:
                if not self.queue:
                    continue

                oldest_item = self.queue[0]
                elapsed = time.perf_counter() - oldest_item.arrival_time

                if len(self.queue) >= self.max_batch_size or elapsed >= self.max_timeout:
                    # Form batch
                    batch_items = self.queue[: self.max_batch_size]
                    self.queue = self.queue[self.max_batch_size :]

                    # Stack inputs
                    batch_inputs = np.vstack([item.input_features for item in batch_items])
                    try:
                        batch_preds = self.scoring_fn(batch_inputs)
                        for idx, item in enumerate(batch_items):
                            if not item.future.done():
                                item.future.set_result(batch_preds[idx])
                    except Exception as e:
                        for item in batch_items:
                            if not item.future.done():
                                item.future.set_exception(e)
