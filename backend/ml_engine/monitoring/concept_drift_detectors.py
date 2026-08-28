"""
ModelForge AI - ML Engine: Concept Drift & Change Point Detection Suite
Implements Bifet & Gavalda Adaptive Windowing (ADWIN), Page-Hinkley test,
Cumulative Sum (CUSUM), and Drift Detection Method (DDM) for streaming classification shifts.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math
import numpy as np


class ADWIN:
    """Adaptive Windowing (ADWIN) algorithm dynamically adjusting window size based on statistically significant rate change."""
    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self.window: List[float] = []
        self.total: float = 0.0
        self.variance: float = 0.0
        self.width: int = 0

    def add_element(self, value: float) -> bool:
        """Add streaming element and check if concept drift occurred."""
        self.window.append(value)
        self.total += value
        self.width += 1
        drift_detected = False

        # Evaluate sub-window splits
        if self.width > 10:
            for split in range(5, self.width - 5):
                w0 = self.window[:split]
                w1 = self.window[split:]
                n0, n1 = len(w0), len(w1)
                mu0 = sum(w0) / n0
                mu1 = sum(w1) / n1

                # Hoeffding bound threshold
                m = 1.0 / (1.0 / n0 + 1.0 / n1)
                eps_cut = math.sqrt((1.0 / (2.0 * m)) * math.log(4.0 * self.width / self.delta))

                if abs(mu0 - mu1) >= eps_cut:
                    drift_detected = True
                    # Shrink window by dropping oldest elements
                    self.window = self.window[split:]
                    self.width = len(self.window)
                    self.total = sum(self.window)
                    break

        return drift_detected


class PageHinkley:
    """Page-Hinkley cumulative sum change detector for abrupt stream mean shifts."""
    def __init__(self, delta: float = 0.005, threshold: float = 50.0, alpha: float = 0.9999):
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.x_mean = 0.0
        self.sample_count = 0
        self.sum_stat = 0.0
        self.min_sum = float("inf")

    def update(self, x: float) -> bool:
        self.sample_count += 1
        self.x_mean = self.alpha * self.x_mean + (1.0 - self.alpha) * x
        self.sum_stat += (x - self.x_mean - self.delta)

        if self.sum_stat < self.min_sum:
            self.min_sum = self.sum_stat

        ph_stat = self.sum_stat - self.min_sum
        if ph_stat > self.threshold:
            # Reset detector
            self.sum_stat = 0.0
            self.min_sum = float("inf")
            return True
        return False


class DriftDetectionMethod:
    """Gama et al. Drift Detection Method (DDM) monitoring online error rates."""
    def __init__(self, min_instances: int = 30):
        self.min_instances = min_instances
        self.sample_count = 0
        self.error_count = 0
        self.p_min = float("inf")
        self.s_min = float("inf")

    def update(self, is_error: bool) -> str:
        self.sample_count += 1
        if is_error:
            self.error_count += 1

        if self.sample_count < self.min_instances:
            return "STABLE"

        p = self.error_count / self.sample_count
        s = math.sqrt(p * (1.0 - p) / self.sample_count)

        if p + s < self.p_min + self.s_min:
            self.p_min = p
            self.s_min = s

        # Warning level: p + s >= p_min + 2.0 * s_min
        if p + s >= self.p_min + 3.0 * self.s_min:
            # Reset
            self.sample_count = 0
            self.error_count = 0
            self.p_min = float("inf")
            self.s_min = float("inf")
            return "DRIFT"
        elif p + s >= self.p_min + 2.0 * self.s_min:
            return "WARNING"

        return "STABLE"
