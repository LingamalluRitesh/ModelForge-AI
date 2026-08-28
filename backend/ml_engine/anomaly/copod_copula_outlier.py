"""
ModelForge AI - ML Engine: Copula-Based Outlier Detection (COPOD)
Implements Li et al. COPOD: Copula-Based Outlier Detection
parameter-free, fast tail probability estimation using empirical cumulative distribution functions.
$O(x) = -\sum_{d=1}^D \left( \log(F_d(x_d)) + \log(1 - F_d(x_d)) ight)$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class COPOD:
    """Copula-Based Parameter-Free Outlier Detector."""
    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination
        self.X_fit_: Optional[np.ndarray] = None
        self.left_cdfs_: List[np.ndarray] = []
        self.right_cdfs_: List[np.ndarray] = []
        self.threshold_: float = 0.0

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        self.X_fit_ = X
        n_samples, n_features = X.shape

        self.left_cdfs_ = []
        self.right_cdfs_ = []

        for j in range(n_features):
            col = X[:, j]
            # Left tail empirical CDF
            ranks_left = np.argsort(np.argsort(col)) + 1
            ecdf_left = ranks_left / float(n_samples)
            # Right tail empirical survival function
            ecdf_right = (n_samples - ranks_left + 1) / float(n_samples)

            self.left_cdfs].append(ecdf_left)
            self.right_cdfs.append(ecdf_right)

        scores = self.score_samples(X)
        self.threshold_ = float(np.quantile(scores, 1.0 - self.contamination))
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        outlier_scores = np.zeros(n_samples)

        for j in range(n_features):
            col_fit = np.sort(self.X_fit_[:, j])
            n_fit = len(col_fit)

            # Map test points to empirical quantiles
            ranks = np.searchsorted(col_fit, X[:, j], side="right")
            p_left = np.clip((ranks + 0.5) / (n_fit + 1.0), 1e-6, 1.0 - 1e-6)
            p_right = np.clip(1.0 - p_left, 1e-6, 1.0 - 1e-6)

            # Tail probability score
            outlier_scores += - (np.log(p_left) + np.log(p_right))

        return outlier_scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.score_samples(X)
        return np.where(scores >= self.threshold_, -1, 1)
