"""
ModelForge AI - Outlier Detection & Robust Clipping
Implements IQR filtering, Z-Score truncation, and Winsorization.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class RobustOutlierHandler:
    """
    Handles numerical outliers via Winsorization, IQR clipping, or Z-Score filtering.
    """

    def __init__(
        self,
        strategy: str = "clip_iqr",  # "clip_iqr", "clip_zscore", "winsorize"
        iqr_multiplier: float = 1.5,
        z_threshold: float = 3.0,
        winsor_percentiles: tuple = (0.01, 0.99),
    ):
        self.strategy = strategy
        self.iqr_multiplier = iqr_multiplier
        self.z_threshold = z_threshold
        self.winsor_percentiles = winsor_percentiles
        self.lower_bounds_: Dict[int, float] = {}
        self.upper_bounds_: Dict[int, float] = {}

    def fit(self, X: np.ndarray):
        n_features = X.shape[1]
        for col_idx in range(n_features):
            col = X[:, col_idx]
            col_clean = col[~np.isnan(col)]

            if len(col_clean) == 0:
                self.lower_bounds_[col_idx] = -np.inf
                self.upper_bounds_[col_idx] = np.inf
                continue

            if self.strategy == "clip_iqr":
                q25, q75 = np.percentile(col_clean, [25, 75])
                iqr = q75 - q25
                self.lower_bounds_[col_idx] = float(q25 - self.iqr_multiplier * iqr)
                self.upper_bounds_[col_idx] = float(q75 + self.iqr_multiplier * iqr)

            elif self.strategy == "clip_zscore":
                mean = np.mean(col_clean)
                std = np.std(col_clean)
                self.lower_bounds_[col_idx] = float(mean - self.z_threshold * std)
                self.upper_bounds_[col_idx] = float(mean + self.z_threshold * std)

            elif self.strategy == "winsorize":
                p_low, p_high = np.percentile(col_clean, [self.winsor_percentiles[0] * 100, self.winsor_percentiles[1] * 100])
                self.lower_bounds_[col_idx] = float(p_low)
                self.upper_bounds_[col_idx] = float(p_high)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_out = X.copy()
        for col_idx, low in self.lower_bounds_.items():
            high = self.upper_bounds_[col_idx]
            X_out[:, col_idx] = np.clip(X_out[:, col_idx], low, high)
        return X_out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
