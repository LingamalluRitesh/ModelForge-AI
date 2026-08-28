"""
ModelForge AI - ML Engine: Hierarchical Time Series Forecast Reconciliation
Implements Hyndman et al. Optimal Forecast Reconciliation (MinT - Minimum Trace),
Bottom-Up (BU), and Top-Down (TD) matrix projections guaranteeing coherent aggregation:
$\tilde{\mathbf{y}}_n = \mathbf{S} (\mathbf{S}^T \mathbf{W}^{-1} \mathbf{S})^{-1} \mathbf{S}^T \mathbf{W}^{-1} \hat{\mathbf{y}}_n$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class HierarchicalForecastReconciler:
    """Optimal Minimum Trace (MinT) hierarchical forecast reconciliation."""

    def __init__(self, summing_matrix: np.ndarray, method: str = "mint_shrink"):
        self.S = np.asarray(summing_matrix, dtype=np.float64)
        self.method = method
        self.n_total, self.n_bottom = self.S.shape
        self.P_: Optional[np.ndarray] = None

    def fit(self, base_forecast_residuals: Optional[np.ndarray] = None):
        """Precompute optimal projection matrix $P = (S^T W^{-1} S)^{-1} S^T W^{-1}$."""
        if self.method == "bottom_up":
            # Bottom-up: extract bottom level directly
            P = np.zeros((self.n_bottom, self.n_total))
            P[:, -self.n_bottom :] = np.eye(self.n_bottom)
            self.P_ = P
            return self

        # Error covariance matrix W
        if base_forecast_residuals is not None and len(base_forecast_residuals) > 0:
            res = np.asarray(base_forecast_residuals, dtype=np.float64)
            W = np.cov(res, rowvar=False)
            # Add shrinkage regularization
            W += 1e-4 * np.eye(self.n_total)
        else:
            # Ordinary least squares projection: W = I
            W = np.eye(self.n_total)

        # Invert covariance
        W_inv = np.linalg.pinv(W)

        # MinT projection: (S^T W^-1 S)^-1 S^T W^-1
        A = np.dot(self.S.T, np.dot(W_inv, self.S))
        A_inv = np.linalg.pinv(A)
        self.P_ = np.dot(A_inv, np.dot(self.S.T, W_inv))
        return self

    def reconcile(self, base_forecasts: np.ndarray) -> np.ndarray:
        """
        Produce coherent hierarchical forecasts satisfying $\tilde{y} = S \tilde{y}_{bottom}$.
        base_forecasts shape: $(H, n\_total)$ or $(n\_total,)$
        """
        y_hat = np.asarray(base_forecasts, dtype=np.float64)
        single_point = y_hat.ndim == 1
        if single_point:
            y_hat = y_hat.reshape(1, -1)

        # Map to bottom level: y_bottom = y_hat * P^T
        y_bottom_reconciled = np.dot(y_hat, self.P_.T)

        # Re-aggregate coherently: y_tilde = y_bottom * S^T
        y_coherent = np.dot(y_bottom_reconciled, self.S.T)

        return y_coherent[0] if single_point else y_coherent
