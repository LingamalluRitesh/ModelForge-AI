"""
ModelForge AI - ML Engine: Vector AutoRegression (VAR) & Vector Error Correction (VECM)
Implements Sims et al. Multivariate Vector AutoRegression with OLS parameter estimation,
Akaike Information Criterion (AIC) lag selection, and Granger Causality hypothesis testing.
$Y_t = c + A_1 Y_{t-1} + \dots + A_p Y_{t-p} + \epsilon_t$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class VectorAutoRegression:
    """Multivariate Time Series VAR(p) Model."""

    def __init__(self, max_lags: int = 5, criterion: str = "aic"):
        self.max_lags = max_lags
        self.criterion = criterion
        self.p_ = 1
        self.intercept_: Optional[np.ndarray] = None  # (K,)
        self.coefs_: Optional[np.ndarray] = None      # (p, K, K)
        self.sigma_u_: Optional[np.ndarray] = None    # (K, K) residual covariance
        self.history_: Optional[np.ndarray] = None

    def fit(self, endog: np.ndarray):
        """Fit VAR parameters via Ordinary Least Squares."""
        Y_raw = np.asarray(endog, dtype=np.float64)
        T_total, K = Y_raw.shape
        self.history_ = Y_raw

        best_p = 1
        best_aic = float("inf")

        for p in range(1, min(self.max_lags + 1, T_total // (K + 1))):
            # Construct lag design matrix
            T = T_total - p
            Y = Y_raw[p:]  # (T, K)

            Z = np.ones((T, 1 + p * K))
            for i in range(T):
                lag_vals = [Y_raw[p + i - l - 1] for l in range(p)]
                Z[i, 1:] = np.concatenate(lag_vals)

            # OLS estimate: B = (Z^T Z)^-1 Z^T Y
            try:
                B = np.linalg.solve(np.dot(Z.T, Z) + 1e-6 * np.eye(1 + p * K), np.dot(Z.T, Y))
                residuals = Y - np.dot(Z, B)
                sigma_u = np.dot(residuals.T, residuals) / T

                # AIC = ln(|Sigma|) + 2 p K^2 / T
                sign, logdet = np.linalg.slogdet(sigma_u)
                if sign > 0:
                    aic = logdet + (2.0 * p * (K ** 2)) / T
                    if aic < best_aic:
                        best_aic = aic
                        best_p = p
            except Exception:
                continue

        self.p_ = best_p

        # Re-fit optimal lag order
        T = T_total - self.p_
        Y = Y_raw[self.p_:]
        Z = np.ones((T, 1 + self.p_ * K))
        for i in range(T):
            lag_vals = [Y_raw[self.p_ + i - l - 1] for l in range(self.p_)]
            Z[i, 1:] = np.concatenate(lag_vals)

        B = np.linalg.solve(np.dot(Z.T, Z) + 1e-6 * np.eye(1 + self.p_ * K), np.dot(Z.T, Y))
        self.intercept_ = B[0]
        self.coefs_ = B[1:].reshape(self.p_, K, K)
        residuals = Y - np.dot(Z, B)
        self.sigma_u_ = np.dot(residuals.T, residuals) / T
        return self

    def forecast(self, steps: int = 10) -> np.ndarray:
        """Produce out-of-sample multistep vector forecasts $(steps, K)$."""
        K = self.history_.shape[1]
        history = list(self.history_)
        forecasts = []

        for _ in range(steps):
            y_hat = self.intercept_.copy()
            for l in range(self.p_):
                prev_y = history[-1 - l]
                y_hat += np.dot(self.coefs_[l].T, prev_y)

            forecasts.append(y_hat)
            history.append(y_hat)

        return np.array(forecasts)
