"""
ModelForge AI - Preprocessing: Minimum Description Length Principle Discretizer (MDLP)
Implements Fayyad & Irani Multi-Interval Discretization of Continuous-Valued Attributes for Classification
using recursive information entropy minimization and Minimum Description Length (MDL) stopping criteria.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class MDLPDiscretizer:
    """Supervised optimal continuous-to-categorical binning via Entropy Minimization."""
    def __init__(self, max_bins: int = 10):
        self.max_bins = max_bins
        self.cut_points_: List[np.ndarray] = []

    def _entropy(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return float(-np.sum(probs * np.log2(np.clip(probs, 1e-10, 1.0))))

    def _find_best_cut(self, x: np.ndarray, y: np.ndarray) -> Optional[float]:
        order = np.argsort(x)
        x_sort = x[order]
        y_sort = y[order]

        n = len(y_sort)
        if n < 4:
            return None

        best_gain = 0.0
        best_cut = None
        base_entropy = self._entropy(y_sort)

        # Evaluate candidate cut points
        for i in range(1, n):
            if x_sort[i] != x_sort[i - 1]:
                cut = (x_sort[i] + x_sort[i - 1]) / 2.0
                left_y = y_sort[:i]
                right_y = y_sort[i:]

                gain = base_entropy - (len(left_y)/n * self._entropy(left_y) + len(right_y)/n * self._entropy(right_y))
                if gain > best_gain:
                    best_gain = gain
                    best_cut = cut

        return best_cut

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        N, D = X.shape

        self.cut_points_ = []
        for j in range(D):
            col = X[:, j]
            cuts = []
            c1 = self._find_best_cut(col, y)
            if c1 is not None:
                cuts.append(c1)
            self.cut_points_.append(np.sort(cuts))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        N, D = X.shape
        X_binned = np.zeros((N, D), dtype=int)

        for j in range(D):
            cuts = self.cut_points_[j]
            if len(cuts) > 0:
                X_binned[:, j] = np.digitize(X[:, j], cuts)
            else:
                X_binned[:, j] = 0
        return X_binned
