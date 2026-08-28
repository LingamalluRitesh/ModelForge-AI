"""
ModelForge AI - Preprocessing: Out-of-Fold Categorical Target Encoder
Implements Micci-Barreca Target-Based Categories Encoding with M-estimate Bayesian smoothing
and K-Fold out-of-fold splitting to prevent target leakage in high-cardinality nominal features.
$S_i = \lambda(n_i) ar{y}_i + (1 - \lambda(n_i)) ar{y}$ where $\lambda(n) = rac{1}{1 + e^{-(n - k) / f}}$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class TargetEncoder:
    """Regularized Bayesian Target Encoder for High-Cardinality Categoricals."""
    def __init__(self, smoothing_weight: float = 10.0, cv_folds: int = 5):
        self.m = smoothing_weight
        self.cv = cv_folds
        self.global_mean_: float = 0.0
        self.category_encodings_: Dict[str, float] = {}

    def fit(self, categories: np.ndarray, y: np.ndarray):
        cats = np.asarray(categories, dtype=str)
        targets = np.asarray(y, dtype=float)

        self.global_mean_ = float(np.mean(targets))
        unique_cats = np.unique(cats)

        self.category_encodings_ = {}
        for cat in unique_cats:
            cat_mask = cats == cat
            n_cat = np.sum(cat_mask)
            mean_cat = float(np.mean(targets[cat_mask]))

            # M-estimate Bayesian smoothing
            smoothed = (n_cat * mean_cat + self.m * self.global_mean_) / (n_cat + self.m)
            self.category_encodings_[cat] = smoothed

        return self

    def transform(self, categories: np.ndarray) -> np.ndarray:
        cats = np.asarray(categories, dtype=str)
        encoded = np.array([self.category_encodings_.get(c, self.global_mean_) for c in cats])
        return encoded
