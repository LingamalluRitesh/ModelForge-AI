"""
ModelForge AI - ML Engine: Prometheus Custom Metrics Exporter
Exposes real-time OpenMetrics / Prometheus scrape endpoints tracking inference request rates,
p50/p90/p95/p99 latency histogram buckets, PSI/KS data drift scores, and GPU memory utilization.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import time
from collections import defaultdict
import threading


class PrometheusMetricsExporter:
    """Thread-safe Prometheus / OpenMetrics metric registry and serializer."""

    def __init__(self):
        self._lock = threading.Lock()
        self._counters: Dict[str, Dict[Tuple[Tuple[str, str], ...], float]] = defaultdict(lambda: defaultdict(float))
        self._gauges: Dict[str, Dict[Tuple[Tuple[str, str], ...], float]] = defaultdict(lambda: defaultdict(float))
        self._histograms: Dict[str, Dict[Tuple[Tuple[str, str], ...], Dict[float, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self._histogram_sums: Dict[str, Dict[Tuple[Tuple[str, str], ...], float]] = defaultdict(lambda: defaultdict(float))
        self._histogram_counts: Dict[str, Dict[Tuple[Tuple[str, str], ...], int]] = defaultdict(lambda: defaultdict(int))

        # Default standard latency buckets (in milliseconds)
        self.DEFAULT_LATENCY_BUCKETS = [1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0]

    def _labels_to_key(self, labels: Optional[Dict[str, str]]) -> Tuple[Tuple[str, str], ...]:
        if not labels:
            return ()
        return tuple(sorted(labels.items()))

    def inc_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment an accumulating Prometheus counter."""
        key = self._labels_to_key(labels)
        with self._lock:
            self._counters[name][key] += value

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set instantaneous Prometheus gauge value."""
        key = self._labels_to_key(labels)
        with self._lock:
            self._gauges[name][key] = value

    def observe_histogram(
        self,
        name: str,
        value: float,
        buckets: Optional[List[float]] = None,
        labels: Optional[Dict[str, str]] = None,
    ):
        """Observe sample in Prometheus histogram distribution buckets."""
        key = self._labels_to_key(labels)
        b_list = sorted(buckets or self.DEFAULT_LATENCY_BUCKETS)
        with self._lock:
            self._histogram_sums[name][key] += value
            self._histogram_counts[name][key] += 1
            for b in b_list:
                if value <= b:
                    self._histograms[name][key][b] += 1

    def generate_openmetrics_payload(self) -> str:
        """Render metrics into standard Prometheus OpenMetrics text format."""
        lines = []

        with self._lock:
            # 1. Counters
            for name, series_map in self._counters.items():
                lines.append(f"# TYPE {name} counter")
                for key, val in series_map.items():
                    label_str = ",".join(f'{k}="{v}"' for k, v in key)
                    labels_formatted = f"{{{label_str}}}" if label_str else ""
                    lines.append(f"{name}{labels_formatted} {val}")

            # 2. Gauges
            for name, series_map in self._gauges.items():
                lines.append(f"# TYPE {name} gauge")
                for key, val in series_map.items():
                    label_str = ",".join(f'{k}="{v}"' for k, v in key)
                    labels_formatted = f"{{{label_str}}}" if label_str else ""
                    lines.append(f"{name}{labels_formatted} {val}")

            # 3. Histograms
            for name, series_map in self._histograms.items():
                lines.append(f"# TYPE {name} histogram")
                for key, b_counts in series_map.items():
                    label_pairs = list(key)
                    cum_count = 0
                    for b in sorted(b_counts.keys()):
                        cum_count += b_counts[b]
                        b_labels = label_pairs + [("le", str(b))]
                        label_str = ",".join(f'{k}="{v}"' for k, v in b_labels)
                        lines.append(f"{name}_bucket{{{label_str}}} {cum_count}")

                    # +Inf bucket
                    inf_labels = label_pairs + [("le", "+Inf")]
                    inf_str = ",".join(f'{k}="{v}"' for k, v in inf_labels)
                    total_count = self._histogram_counts[name][key]
                    lines.append(f"{name}_bucket{{{inf_str}}} {total_count}")

                    # Sum and Count
                    base_str = ",".join(f'{k}="{v}"' for k, v in label_pairs)
                    base_formatted = f"{{{base_str}}}" if base_str else ""
                    lines.append(f"{name}_sum{base_formatted} {self._histogram_sums[name][key]}")
                    lines.append(f"{name}_count{base_formatted} {total_count}")

        return "\n".join(lines) + "\n"


# Global singleton exporter
prometheus_exporter = PrometheusMetricsExporter()
