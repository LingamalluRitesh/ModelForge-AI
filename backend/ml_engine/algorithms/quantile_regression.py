"""
ModelForge AI - ML Engine: Quantile Regression & Conformal Prediction Intervals
Implements Pinball loss optimization, Asymmetric Laplace Likelihood regression,
and Split-Conformal prediction sets with distribution-free finite-sample coverage guarantees.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import minimize


class QuantileRegressor:
    """Linear Quantile Regression via Pinball Loss minimization: $\mathcal{L}_q(y, \hat{y}) = \max(q(y - \hat{y}), (q - 1)(y - \hat{y}))$."""

    def __init__(self, quantile: float = 0.5, fit_intercept: bool = True, alpha: float = 1e-4):
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"Quantile must be in (0, 1), got {quantile}")
        self.quantile = quantile
        self.fit_intercept = fit_intercept
        self.alpha = alpha
        self.weights_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def _pinball_loss(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        if self.fit_intercept:
            w = params[:-1]
            b = params[-1]
            y_pred = np.dot(X, w) + b
        else:
            w = params
            y_pred = np.dot(X, w)

        residuals = y - y_pred
        loss = np.where(residuals >= 0, self.quantile * residuals, (self.quantile - 1.0) * residuals)
        l2_reg = 0.5 * self.alpha * np.sum(w ** 2)
        return float(np.mean(loss) + l2_reg)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_features = X.shape[1]

        init_params = np.zeros(n_features + (1 if self.fit_intercept else 0))
        res = minimize(
            self._pinball_loss,
            init_params,
            args=(X, y),
            method="BFGS",
            options={"maxiter": 500, "disp": False},
        )

        if self.fit_intercept:
            self.weights_ = res.x[:-1]
            self.intercept_ = float(res.x[-1])
        else:
            self.weights_ = res.x
            self.intercept_ = 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        y_pred = np.dot(X, self.weights_) + self.intercept_
        return y_pred


class ConformalPredictionCalibrator:
    """
    Split Conformal Prediction providing valid coverage intervals:
    $P(Y_{n+1} \in \hat{C}(X_{n+1})) \ge 1 - \alpha$
    """

    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self.conformity_scores_: Optional[np.ndarray] = None
        self.quantile_threshold_: float = 0.0

    def calibrate(self, y_true_cal: np.ndarray, y_pred_cal: np.ndarray):
        """Compute non-conformity residuals on hold-out calibration dataset."""
        y_true = np.asarray(y_true_cal, dtype=np.float64)
        y_pred = np.asarray(y_pred_cal, dtype=np.float64)
        residuals = np.abs(y_true - y_pred)
        self.conformity_scores_ = np.sort(residuals)

        n = len(self.conformity_scores_)
        # Conformal adjusted quantile index: ceil((n + 1)(1 - alpha)) / n
        q_level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.quantile_threshold_ = float(np.quantile(self.conformity_scores_, q_level, method="higher"))
        return self

    def predict_interval(self, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate prediction intervals [lower_bound, upper_bound]."""
        y_p = np.asarray(y_pred, dtype=np.float64)
        lower = y_p - self.quantile_threshold_
        upper = y_p + self.quantile_threshold_
        return lower, upper
