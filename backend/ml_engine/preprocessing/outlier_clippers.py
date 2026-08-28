"""
ModelForge AI - Preprocessing: Robust Outlier Clippers & Median Absolute Deviation (MAD) Scaler
Implements Hampel filter, Winsorization, and Robust Median Absolute Deviation scaling
for extreme noise resistance in dirty tabular streams.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class Winsorizer:
    """Clips extreme tails to predetermined lower and upper percentile boundaries."""
    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_q = lower_quantile
        self.upper_q = upper_quantile
        self.lower_bounds_: Optional[np.ndarray] = None
        self.upper_bounds_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.quantile(X, self.lower_q, axis=0)
        self.upper_bounds_ = np.quantile(X, self.upper_q, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)


class RobustMADScaler:
    """Standardizes features using Median and Median Absolute Deviation: $x_{scaled} = rac{x - 	ext{median}}{1.4826 	imes 	ext{MAD}}$."""
    def __init__(self):
        self.medians_: Optional[np.ndarray] = None
        self.mads_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        self.medians_ = np.median(X, axis=0)
        deviations = np.abs(X - self.medians_)
        # Normal distribution consistency constant 1.4826
        self.mads_ = 1.4826 * np.median(deviations, axis=0)
        self.mads_ = np.maximum(1e-8, self.mads_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return (X - self.medians_) / self.mads_
