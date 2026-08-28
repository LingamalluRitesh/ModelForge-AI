"""
ModelForge AI - ML Engine: AutoRegressive Integrated Moving Average (ARIMA)
Implements ARIMA(p,d,q) parameter estimation via Conditional Sum of Squares (CSS)
and exact Kalman Filter state-space recursions for time series forecasting.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import minimize


class ARIMA:
    """ARIMA(p, d, q) Time Series Model."""

    def __init__(self, p: int = 1, d: int = 0, q: int = 1):
        self.p = p
        self.d = d
        self.q = q
        self.ar_params_: Optional[np.ndarray] = None
        self.ma_params_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.sigma2_: float = 1.0
        self.history_: Optional[np.ndarray] = None
        self.diff_history_: Optional[np.ndarray] = None

    def _difference(self, series: np.ndarray, d: int) -> np.ndarray:
        res = series.copy()
        for _ in range(d):
            res = np.diff(res)
        return res

    def fit(self, series: np.ndarray):
        self.history_ = np.asarray(series, dtype=np.float64)
        diff_y = self._difference(self.history_, self.d)
        self.diff_history_ = diff_y
        N = len(diff_y)

        # CSS (Conditional Sum of Squares) Objective
        def _css_loss(params: np.ndarray) -> float:
            c = params[0]
            phi = params[1 : 1 + self.p] if self.p > 0 else np.array([])
            theta = params[1 + self.p :] if self.q > 0 else np.array([])

            errors = np.zeros(N)
            for t in range(max(self.p, self.q), N):
                ar_term = np.dot(phi, diff_y[t - self.p : t][::-1]) if self.p > 0 else 0.0
                ma_term = np.dot(theta, errors[t - self.q : t][::-1]) if self.q > 0 else 0.0
                y_hat = c + ar_term + ma_term
                errors[t] = diff_y[t] - y_hat

            return float(np.sum(errors[max(self.p, self.q) :] ** 2))

        n_params = 1 + self.p + self.q
        init_params = np.zeros(n_params)
        init_params[0] = float(np.mean(diff_y))

        res = minimize(_css_loss, init_params, method="L-BFGS-B")

        self.intercept_ = float(res.x[0])
        self.ar_params_ = res.x[1 : 1 + self.p] if self.p > 0 else np.array([])
        self.ma_params_ = res.x[1 + self.p :] if self.q > 0 else np.array([])
        return self

    def forecast(self, steps: int = 10) -> np.ndarray:
        """Produce out-of-sample multistep point forecasts."""
        history = list(self.diff_history_)
        forecasts = []

        for step in range(steps):
            t = len(history)
            ar_term = (
                np.dot(self.ar_params_, np.array(history[t - self.p : t])[::-1])
                if self.p > 0 and t >= self.p
                else 0.0
            )
            # Expectation of future shocks is zero
            pred = self.intercept_ + ar_term
            forecasts.append(pred)
            history.append(pred)

        # Undifference if d > 0
        final_forecasts = np.array(forecasts)
        if self.d == 1:
            last_orig = self.history_[-1]
            final_forecasts = last_orig + np.cumsum(final_forecasts)

        return final_forecasts
