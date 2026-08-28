"""
ModelForge AI - ML Engine: Conformal Prediction Uncertainty Bounds
Implements Vovk, Gammerman, & Shafer Algorithmic Learning in a Random World (Conformal Prediction)
providing finite-sample distribution-free coverage guarantees $P(y \in C(x)) \ge 1 - lpha$.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class ConformalPredictor:
    """Split-conformal prediction intervals for regression and classification."""
    def __init__(self, confidence_level: float = 0.90):
        self.alpha = 1.0 - confidence_level
        self.q_hat_: float = 0.0

    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        n = len(y_true)

        # Non-conformity scores: absolute residuals
        residuals = np.abs(y_true - y_pred)

        # Finite-sample adjusted quantile: ceil((n + 1) * (1 - alpha)) / n
        q_level = np.ceil((n + 1.0) * (1.0 - self.alpha)) / float(n)
        q_level = np.clip(q_level, 0.0, 1.0)
        self.q_hat_ = float(np.quantile(residuals, q_level))
        return self

    def predict_interval(self, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y_pred = np.asarray(y_pred, dtype=float)
        lower_bound = y_pred - self.q_hat_
        upper_bound = y_pred + self.q_hat_
        return lower_bound, upper_bound
