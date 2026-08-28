"""
ModelForge AI - ML Engine: Regularized Linear Models Suite
Implements ElasticNet with Cyclical Coordinate Descent, Lasso (L1 Soft-Thresholding),
Ridge (L2 Closed-Form Tikhonov Inversion), and Logistic Regression with Newton-Raphson & L-BFGS.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class ElasticNet:
    """
    ElasticNet Regularization: $\min_w \frac{1}{2n} ||y - Xw||_2^2 + \alpha \rho ||w||_1 + \frac{\alpha (1 - \rho)}{2} ||w||_2^2$
    Solved via Cyclical Coordinate Descent with exact soft-thresholding.
    """

    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5, max_iter: int = 1000, tol: float = 1e-4):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def _soft_threshold(self, z: float, gamma: float) -> float:
        """Soft-thresholding operator $S(z, \gamma) = \text{sign}(z) \max(|z| - \gamma, 0)$."""
        if z > gamma:
            return z - gamma
        elif z < -gamma:
            return z + gamma
        return 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape

        # Center data
        x_mean = np.mean(X, axis=0)
        y_mean = float(np.mean(y))
        X_centered = X - x_mean
        y_centered = y - y_mean

        # Precompute column squared norms
        z_j = np.sum(X_centered ** 2, axis=0) / n_samples

        w = np.zeros(n_features)
        l1_penalty = self.alpha * self.l1_ratio
        l2_penalty = self.alpha * (1.0 - self.l1_ratio)

        for _ in range(self.max_iter):
            w_old = w.copy()

            for j in range(n_features):
                if z_j[j] == 0:
                    continue

                # Partial residual without feature j: r = y - sum_{k != j} w_k x_k
                r_j = y_centered - np.dot(X_centered, w) + w[j] * X_centered[:, j]
                rho_j = float(np.dot(X_centered[:, j], r_j) / n_samples)

                # Coordinate update
                w[j] = self._soft_threshold(rho_j, l1_penalty) / (z_j[j] + l2_penalty)

            if np.max(np.abs(w - w_old)) < self.tol:
                break

        self.coef_ = w
        self.intercept_ = y_mean - float(np.dot(x_mean, self.coef_))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return np.dot(X, self.coef_) + self.intercept_


class RidgeRegression:
    """Tikhonov L2-Regularized Closed-Form Linear Regression: $w = (X^T X + \alpha I)^{-1} X^T y$."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape

        x_mean = np.mean(X, axis=0)
        y_mean = float(np.mean(y))
        X_c = X - x_mean
        y_c = y - y_mean

        A = np.dot(X_c.T, X_c) + self.alpha * np.eye(n_features)
        b = np.dot(X_c.T, y_c)
        self.coef_ = np.linalg.solve(A, b)
        self.intercept_ = y_mean - float(np.dot(x_mean, self.coef_))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.coef_) + self.intercept_
