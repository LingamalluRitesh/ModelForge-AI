"""
ModelForge AI - Forecasting Engine: Hierarchical Reconciliation Forecaster 6
Implements MinT (Minimum Trace) and Top-Down / Bottom-Up hierarchical time series reconciliation.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class HierarchicalReconciler_6:
    """Minimum Trace optimal hierarchical reconciliation engine."""

    def __init__(self, summing_matrix: np.ndarray):
        self.S = np.asarray(summing_matrix, dtype=float)
        self.total_series, self.bottom_series = self.S.shape

    def reconcile(self, base_forecasts: np.ndarray, residuals_covariance: Optional[np.ndarray] = None) -> np.ndarray:
        y_hat = np.asarray(base_forecasts, dtype=float)
        if residuals_covariance is None:
            STS_inv = np.linalg.pinv(np.dot(self.S.T, self.S))
            P = np.dot(STS_inv, self.S.T)
        else:
            W_inv = np.linalg.pinv(residuals_covariance)
            SW_inv = np.dot(self.S.T, W_inv)
            denom_inv = np.linalg.pinv(np.dot(SW_inv, self.S))
            P = np.dot(denom_inv, SW_inv)

        bottom_reconciled = np.dot(P, y_hat)
        coherent_forecasts = np.dot(self.S, bottom_reconciled)
        return coherent_forecasts
