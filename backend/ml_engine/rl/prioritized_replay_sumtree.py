"""
ModelForge AI - RL Engine: Binary SumTree for Proportional Prioritized Experience Replay
Implements Schaul et al. Prioritized Experience Replay
with $O(\log N)$ priority updates and $O(\log N)$ proportional probability transition sampling.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class BinarySumTree:
    """Binary tree where parent node value is the exact sum of its children nodes."""
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        # Tree array size: 2 * capacity - 1
        self.tree = np.zeros(2 * capacity - 1)
        # Data array storing transition payloads
        self.data = [None] * capacity
        self.write_ptr = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total_priority(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data: Any):
        tree_idx = self.write_ptr + self.capacity - 1
        self.data[self.write_ptr] = data
        self.update(tree_idx, priority)

        self.write_ptr = (self.write_ptr + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, tree_idx: int, priority: float):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def get(self, s: float) -> Tuple[int, float, Any]:
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, float(self.tree[idx]), self.data[data_idx]
