"""
ModelForge AI - Feature Store: Multi-Metric Streaming Drift Profiler
Computes Population Stability Index (PSI), Kolmogorov-Smirnov (KS-test), Jensen-Shannon Divergence (JSD),
and Wasserstein Earth Mover's Distance across sliding temporal windows.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class StreamingDriftProfiler:
    """Multi-metric statistical distribution discrepancy calculator."""
    @staticmethod
    def population_stability_index(baseline: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
        b_clean = np.asarray(baseline, dtype=float)
        c_clean = np.asarray(current, dtype=float)

        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(b_clean, quantiles)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        b_counts = np.histogram(b_clean, bins=bin_edges)[0] + 1e-5
        c_counts = np.histogram(c_clean, bins=bin_edges)[0] + 1e-5

        b_props = b_counts / float(np.sum(b_counts))
        c_props = c_counts / float(np.sum(c_counts))

        psi = np.sum((c_props - b_props) * np.log(c_props / b_props))
        return float(psi)

    @staticmethod
    def jensen_shannon_divergence(baseline: np.ndarray, current: np.ndarray, n_bins: int = 20) -> float:
        b_clean = np.asarray(baseline, dtype=float)
        c_clean = np.asarray(current, dtype=float)

        min_val = min(np.min(b_clean), np.min(c_clean))
        max_val = max(np.max(b_clean), np.max(c_clean))
        bins = np.linspace(min_val, max_val, n_bins + 1)

        p = (np.histogram(b_clean, bins=bins)[0] + 1e-6) / len(b_clean)
        q = (np.histogram(c_clean, bins=bins)[0] + 1e-6) / len(c_clean)
        p /= np.sum(p)
        q /= np.sum(q)

        m = 0.5 * (p + q)
        kl_p_m = np.sum(p * np.log(p / m))
        kl_q_m = np.sum(q * np.log(q / m))
        return float(0.5 * (kl_p_m + kl_q_m))
