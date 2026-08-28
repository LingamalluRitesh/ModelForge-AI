"""
ModelForge AI - ML Engine: Double Machine Learning Suite (DoubleML)
Implements Chernozhukov et al. Double/Debiased Machine Learning for Treatment and Structural Parameters
with Neyman-orthogonal score functions, K-Fold cross-fitting, Partially Linear Regression (PLR),
and Interactive Regression Models (IRM).
$Y - \ell(X) = \theta (D - m(X)) + \epsilon$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class DoubleMLPLR:
    """Partially Linear Regression (PLR) with Neyman-Orthogonal score."""

    def __init__(self, n_folds: int = 5):
        self.n_folds = n_folds
        self.coef_: float = 0.0
        self.se_: float = 0.0
        self.t_stat_: float = 0.0
        self.p_value_: float = 0.0
        self.ci_lower_: float = 0.0
        self.ci_upper_: float = 0.0

    def fit(self, X: np.ndarray, d: np.ndarray, y: np.ndarray):
        """
        Estimate treatment coefficient $\theta_0$:
        $\hat{\theta} = \frac{\frac{1}{N} \sum_{i} (D_i - \hat{m}(X_i)) (Y_i - \hat{\ell}(X_i))}{\frac{1}{N} \sum_i (D_i - \hat{m}(X_i))^2}$
        """
        X = np.asarray(X, dtype=np.float64)
        d = np.asarray(d, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        N = len(y)

        # Cross-fitting predictions
        y_hat = np.zeros(N)
        d_hat = np.zeros(N)

        indices = np.arange(N)
        folds = np.array_split(np.random.permutation(indices), self.n_folds)

        for fold in folds:
            train_idx = np.setdiff1d(indices, fold)
            test_idx = fold

            # Nuisance regression 1: l(X) = E[Y|X] via Ridge
            A_y = np.dot(X[train_idx].T, X[train_idx]) + 1.0 * np.eye(X.shape[1])
            b_y = np.dot(X[train_idx].T, y[train_idx])
            w_y = np.linalg.solve(A_y, b_y)
            y_hat[test_idx] = np.dot(X[test_idx], w_y)

            # Nuisance regression 2: m(X) = E[D|X] via Ridge
            A_d = np.dot(X[train_idx].T, X[train_idx]) + 1.0 * np.eye(X.shape[1])
            b_d = np.dot(X[train_idx].T, d[train_idx])
            w_d = np.linalg.solve(A_d, b_d)
            d_hat[test_idx] = np.dot(X[test_idx], w_d)

        # Orthogonalized residuals
        u_hat = y - y_hat
        v_hat = d - d_hat

        # Estimate theta
        num = np.mean(v_hat * u_hat)
        denom = np.mean(v_hat ** 2)
        self.coef_ = float(num / max(1e-10, denom))

        # Asymptotic variance estimator
        psi = (u_hat - self.coef_ * v_hat) * v_hat
        var_theta = np.mean(psi ** 2) / (denom ** 2)
        self.se_ = float(np.sqrt(max(1e-10, var_theta / N)))

        self.t_stat_ = self.coef_ / max(1e-10, self.se_)
        self.ci_lower_ = self.coef_ - 1.96 * self.se_
        self.ci_upper_ = self.coef_ + 1.96 * self.se_

        return self
