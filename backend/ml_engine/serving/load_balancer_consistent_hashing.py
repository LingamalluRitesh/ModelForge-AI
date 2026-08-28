"""
ModelForge AI - Serving: Ketama Consistent Hashing Load Balancer
Provides deterministic, minimal-churn distributed inference routing across dynamic server replica sets.
"""

from typing import Any, Dict, List, Optional
import bisect
import hashlib


class KetamaHashRing:
    """Ketama consistent hash ring with 160 points per physical host."""
    def __init__(self, replicas_per_node: int = 160):
        self.replicas = replicas_per_node
        self.ring: Dict[int, str] = {}
        self.sorted_keys: List[int] = []

    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)

    def add_node(self, node_id: str):
        for i in range(self.replicas):
            k = f"{node_id}:{i}"
            h = self._hash(k)
            self.ring[h] = node_id
            bisect.insort(self.sorted_keys, h)

    def remove_node(self, node_id: str):
        for i in range(self.replicas):
            k = f"{node_id}:{i}"
            h = self._hash(k)
            if h in self.ring:
                del self.ring[h]
                self.sorted_keys.remove(h)

    def get_node(self, partition_key: str) -> Optional[str]:
        if not self.ring:
            return None
        h = self._hash(partition_key)
        idx = bisect.bisect_right(self.sorted_keys, h)
        if idx == len(self.sorted_keys):
            idx = 0
        return self.ring[self.sorted_keys[idx]]
