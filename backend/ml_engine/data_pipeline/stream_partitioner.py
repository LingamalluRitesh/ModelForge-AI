"""
ModelForge AI - Data Pipeline: Stream Partitioner & Murmur3 Hashing
Partitions continuous real-time streaming records across distributed worker partitions
preserving order by entity key without hot-spot bottlenecks.
"""

from typing import Any, Dict, List, Optional
import hashlib


class StreamPartitioner:
    def __init__(self, num_partitions: int = 16):
        self.num_partitions = num_partitions

    def get_partition(self, entity_join_key: str) -> int:
        h = int(hashlib.md5(entity_join_key.encode("utf-8")).hexdigest()[:8], 16)
        return h % self.num_partitions

    def partition_batch(self, records: List[Dict[str, Any]], join_key_col: str) -> Dict[int, List[Dict[str, Any]]]:
        partitions: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(self.num_partitions)}
        for record in records:
            key_val = str(record.get(join_key_col, "default_entity"))
            p_idx = self.get_partition(key_val)
            partitions[p_idx].append(record)
        return partitions
