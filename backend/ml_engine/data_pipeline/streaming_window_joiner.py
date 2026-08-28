"""
ModelForge AI - Data Pipeline: Bi-Temporal Streaming As-Of Join Engine
Executes point-in-time joins between fast event streams and slow feature update streams
guaranteeing strict $t_{feature} \le t_{event}$ order without future information leakage.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import bisect


class StreamingAsOfJoiner:
    """Maintains stateful historical feature queues per entity key."""
    def __init__(self, max_history_per_key: int = 1000):
        self.max_history = max_history_per_key
        # entity_key -> (sorted_timestamps, sorted_records)
        self.feature_history: Dict[str, Tuple[List[float], List[Dict[str, Any]]]] = {}

    def insert_feature_snapshot(self, entity_key: str, timestamp: float, feature_dict: Dict[str, Any]):
        if entity_key not in self.feature_history:
            self.feature_history[entity_key] = ([], [])

        timestamps, records = self.feature_history[entity_key]
        idx = bisect.bisect_right(timestamps, timestamp)
        timestamps.insert(idx, timestamp)
        records.insert(idx, feature_dict)

        if len(timestamps) > self.max_history:
            timestamps.pop(0)
            records.pop(0)

    def join_event(self, entity_key: str, event_timestamp: float) -> Optional[Dict[str, Any]]:
        """Retrieve latest feature snapshot strictly on or before event_timestamp."""
        if entity_key not in self.feature_history:
            return None

        timestamps, records = self.feature_history[entity_key]
        idx = bisect.bisect_right(timestamps, event_timestamp) - 1
        if idx >= 0:
            return records[idx]
        return None
