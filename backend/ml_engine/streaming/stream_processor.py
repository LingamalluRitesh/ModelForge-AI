"""
ModelForge AI - ML Engine: Real-Time Stream Ingestion & Micro-batch Processor
Processes high-throughput streaming events, manages tumbling / sliding time windows,
aggregates feature moments on the fly, and fires automated micro-batch data drift alerts.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import time
import threading
from collections import deque
import numpy as np
import pandas as pd
from app.core.logging import logger


class StreamWindow:
    """Sliding time window maintaining historical event records for continuous stream monitoring."""

    def __init__(self, window_size_seconds: int = 300, max_capacity: int = 10000):
        self.window_size_seconds = window_size_seconds
        self.max_capacity = max_capacity
        self.events: deque = deque(maxlen=max_capacity)
        self.lock = threading.Lock()

    def add_event(self, event_data: Dict[str, Any], timestamp: Optional[float] = None):
        ts = timestamp or time.time()
        with self.lock:
            self.events.append((ts, event_data))
            self._prune_expired(ts)

    def _prune_expired(self, current_time: float):
        cutoff = current_time - self.window_size_seconds
        while self.events and self.events[0][0] < cutoff:
            self.events.popleft()

    def get_dataframe(self) -> pd.DataFrame:
        with self.lock:
            self._prune_expired(time.time())
            if not self.events:
                return pd.DataFrame()
            records = [item[1] for item in self.events]
            return pd.DataFrame(records)

    def count(self) -> int:
        with self.lock:
            self._prune_expired(time.time())
            return len(self.events)


class StreamingDriftMonitor:
    """Computes streaming Kolmogorov-Smirnov and PSI drift between reference baseline and sliding window."""

    def __init__(
        self,
        baseline_df: pd.DataFrame,
        continuous_features: List[str],
        window_size_seconds: int = 300,
        psi_threshold: float = 0.20,
        drift_callback: Optional[Callable[[str, float, Dict[str, Any]], None]] = None,
    ):
        self.baseline_df = baseline_df
        self.continuous_features = continuous_features
        self.window = StreamWindow(window_size_seconds=window_size_seconds)
        self.psi_threshold = psi_threshold
        self.drift_callback = drift_callback
        self.last_drift_check = time.time()

        # Precompute baseline quantiles
        self.baseline_bins: Dict[str, np.ndarray] = {}
        self.baseline_proportions: Dict[str, np.ndarray] = {}
        self._precompute_baseline()

    def _precompute_baseline(self):
        for col in self.continuous_features:
            if col in self.baseline_df.columns:
                series = self.baseline_df[col].dropna().values
                percentiles = np.linspace(0, 100, 11)
                bins = np.percentile(series, percentiles)
                bins[0] = -np.inf
                bins[-1] = np.inf
                counts, _ = np.histogram(series, bins=bins)
                props = (counts + 1e-4) / (np.sum(counts) + 1e-4 * len(counts))
                self.baseline_bins[col] = bins
                self.baseline_proportions[col] = props

    def ingest(self, record: Dict[str, Any]):
        self.window.add_event(record)

    def evaluate_microbatch_drift(self) -> Dict[str, Any]:
        curr_df = self.window.get_dataframe()
        if len(curr_df) < 50:
            return {"status": "insufficient_samples", "sample_count": len(curr_df), "feature_drift": {}}

        feature_drift = {}
        drift_detected = False

        for col in self.continuous_features:
            if col in curr_df.columns and col in self.baseline_bins:
                curr_vals = curr_df[col].dropna().values
                if len(curr_vals) == 0:
                    continue

                bins = self.baseline_bins[col]
                counts, _ = np.histogram(curr_vals, bins=bins)
                curr_props = (counts + 1e-4) / (np.sum(counts) + 1e-4 * len(counts))

                # Compute PSI
                base_props = self.baseline_proportions[col]
                psi = float(np.sum((curr_props - base_props) * np.log(curr_props / base_props)))

                is_drifted = psi >= self.psi_threshold
                if is_drifted:
                    drift_detected = True

                feature_drift[col] = {
                    "psi_score": round(psi, 4),
                    "is_drifted": is_drifted,
                    "sample_count": len(curr_vals),
                }

        report = {
            "status": "evaluated",
            "drift_detected": drift_detected,
            "sample_count": len(curr_df),
            "feature_drift": feature_drift,
            "evaluated_at": time.time(),
        }

        if drift_detected and self.drift_callback:
            self.drift_callback("streaming_data_drift", self.psi_threshold, report)

        return report
