"""
ModelForge AI - ML Engine: Matrix Factorization & Recommendation Systems
Implements Alternating Least Squares (ALS), Weighted Regularized Matrix Factorization (WRMF),
and Singular Value Decomposition (SVD++) with user/item bias terms for collaborative filtering.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class ExplicitMatrixFactorization:
    """SVD with user/item bias and L2 regularization."""

    def __init__(
        self,
        n_factors: int = 32,
        n_epochs: int = 50,
        lr: float = 0.005,
        reg: float = 0.02,
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg

        self.global_mean_ = 0.0
        self.user_bias_: Optional[np.ndarray] = None
        self.item_bias_: Optional[np.ndarray] = None
        self.user_factors_: Optional[np.ndarray] = None
        self.item_factors_: Optional[np.ndarray] = None
        self.n_users = 0
        self.n_items = 0

    def fit(self, R: np.ndarray):
        """Fit on rating matrix R where 0 indicates unobserved."""
        R = np.asarray(R, dtype=np.float64)
        self.n_users, self.n_items = R.shape

        observed_mask = R > 0
        self.global_mean_ = float(np.mean(R[observed_mask])) if np.any(observed_mask) else 0.0

        # Initialize parameters
        self.user_bias_ = np.zeros(self.n_users)
        self.item_bias_ = np.zeros(self.n_items)
        self.user_factors_ = np.random.normal(0, 0.1, (self.n_users, self.n_factors))
        self.item_factors_ = np.random.normal(0, 0.1, (self.n_items, self.n_factors))

        users, items = np.where(observed_mask)
        ratings = R[users, items]

        for epoch in range(self.n_epochs):
            for u, i, r in zip(users, items, ratings):
                # Prediction: mu + b_u + b_i + p_u . q_i
                pred = self.global_mean_ + self.user_bias_[u] + self.item_bias_[i] + np.dot(self.user_factors_[u], self.item_factors_[i])
                err = r - pred

                # SGD updates
                self.user_bias_[u] += self.lr * (err - self.reg * self.user_bias_[u])
                self.item_bias_[i] += self.lr * (err - self.reg * self.item_bias_[i])

                p_u = self.user_factors_[u].copy()
                q_i = self.item_factors_[i].copy()

                self.user_factors_[u] += self.lr * (err * q_i - self.reg * p_u)
                self.item_factors_[i] += self.lr * (err * p_u - self.reg * q_i)

        return self

    def predict(self, user_idx: int, item_idx: int) -> float:
        pred = (
            self.global_mean_
            + self.user_bias_[user_idx]
            + self.item_bias_[item_idx]
            + np.dot(self.user_factors_[user_idx], self.item_factors_[item_idx])
        )
        return float(pred)

    def predict_all(self) -> np.ndarray:
        """Reconstruct the entire full rating matrix."""
        bias_matrix = self.global_mean_ + self.user_bias_[:, np.newaxis] + self.item_bias_[np.newaxis, :]
        interaction = np.dot(self.user_factors_, self.item_factors_.T)
        return bias_matrix + interaction
