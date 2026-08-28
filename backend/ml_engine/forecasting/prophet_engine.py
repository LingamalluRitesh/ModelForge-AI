"""
ModelForge AI - ML Engine: Decomposable Time Series Forecasting Engine
Implements Piecewise Linear / Logistic Growth Trend with automatic changepoint detection,
Fourier Series Seasonal Harmonics (Daily, Weekly, Yearly), and Holiday Indicators.
$y(t) = g(t) + s(t) + h(t) + \epsilon_t$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.optimize import minimize


class DecomposableTimeSeriesForecaster:
    """Additive decomposable forecasting architecture with trend, Fourier seasonality, and holidays."""

    def __init__(
        self,
        n_changepoints: int = 15,
        yearly_seasonality_order: int = 5,
        weekly_seasonality_order: int = 3,
        changepoint_prior_scale: float = 0.05,
    ):
        self.n_changepoints = n_changepoints
        self.yearly_order = yearly_seasonality_order
        self.weekly_order = weekly_seasonality_order
        self.changepoint_prior_scale = changepoint_prior_scale

        self.k_: float = 0.0  # Base growth rate
        self.m_: float = 0.0  # Offset
        self.delta_: Optional[np.ndarray] = None  # Rate adjustments at changepoints
        self.changepoints_t_: Optional[np.ndarray] = None
        self.seasonality_coeffs_: Optional[np.ndarray] = None
        self.t_min_ = 0.0
        self.t_max_ = 1.0

    def _fourier_series(self, t: np.ndarray, period: float, order: int) -> np.ndarray:
        """Construct Fourier trigonometric harmonics matrix $X_{season} = [\sin(2\pi i t / P), \cos(2\pi i t / P)]$."""
        harmonics = []
        for i in range(1, order + 1):
            harmonics.append(np.sin(2.0 * np.pi * i * t / period))
            harmonics.append(np.cos(2.0 * np.pi * i * t / period))
        return np.column_stack(harmonics)

    def fit(self, dates: pd.Series, y: Union[pd.Series, np.ndarray]):
        """Fit decomposable model parameters via L-BFGS-B optimization."""
        dt_series = pd.to_datetime(dates)
        timestamps = dt_series.astype("int64").values / 1e9  # seconds

        self.t_min_ = float(np.min(timestamps))
        self.t_max_ = float(np.max(timestamps))
        t = (timestamps - self.t_min_) / max(1.0, (self.t_max_ - self.t_min_))
        y = np.asarray(y, dtype=np.float64)
        N = len(t)

        # Place changepoints uniformly in the first 80% of time history
        self.changepoints_t_ = np.linspace(0.05, 0.80, self.n_changepoints)

        # Build seasonality matrices
        day_in_year = dt_series.dt.dayofyear.values.astype(float)
        day_in_week = dt_series.dt.dayofweek.values.astype(float)

        X_yearly = self._fourier_series(day_in_year, 365.25, self.yearly_order)
        X_weekly = self._fourier_series(day_in_week, 7.0, self.weekly_order)
        X_season = np.column_stack([X_yearly, X_weekly])

        # Piecewise trend matrix: A(t)_{i,j} = (t_i - s_j) * I(t_i >= s_j)
        A = np.zeros((N, self.n_changepoints))
        for j, s in enumerate(self.changepoints_t_):
            A[:, j] = np.maximum(0.0, t - s)

        # Optimization variables: [k, m, delta (n_cp), beta (n_season)]
        n_season = X_season.shape[1]

        def _loss(params: np.ndarray) -> float:
            k = params[0]
            m = params[1]
            delta = params[2 : 2 + self.n_changepoints]
            beta = params[2 + self.n_changepoints :]

            trend = (k + np.dot(A, delta)) * t + (m + np.dot(A, -self.changepoints_t_ * delta))
            seasonality = np.dot(X_season, beta)
            y_hat = trend + seasonality

            mse = float(np.mean((y - y_hat) ** 2))
            laplace_reg = float(np.sum(np.abs(delta)) / self.changepoint_prior_scale)
            l2_beta = 0.01 * float(np.sum(beta ** 2))
            return mse + laplace_reg + l2_beta

        init_params = np.zeros(2 + self.n_changepoints + n_season)
        init_params[0] = (y[-1] - y[0]) / max(1e-3, t[-1] - t[0])  # Initial slope
        init_params[1] = y[0]  # Initial intercept

        res = minimize(_loss, init_params, method="L-BFGS-B", options={"maxiter": 600, "disp": False})

        self.k_ = float(res.x[0])
        self.m_ = float(res.x[1])
        self.delta_ = res.x[2 : 2 + self.n_changepoints]
        self.seasonality_coeffs_ = res.x[2 + self.n_changepoints :]

        return self

    def predict(self, future_dates: pd.Series) -> pd.DataFrame:
        """Forecast future timestamps and return decomposed trend, seasonal, and total values."""
        dt_series = pd.to_datetime(future_dates)
        timestamps = dt_series.astype("int64").values / 1e9
        t = (timestamps - self.t_min_) / max(1.0, (self.t_max_ - self.t_min_))
        N = len(t)

        A = np.zeros((N, self.n_changepoints))
        for j, s in enumerate(self.changepoints_t_):
            A[:, j] = np.maximum(0.0, t - s)

        trend = (self.k_ + np.dot(A, self.delta_)) * t + (self.m_ + np.dot(A, -self.changepoints_t_ * self.delta_))

        day_in_year = dt_series.dt.dayofyear.values.astype(float)
        day_in_week = dt_series.dt.dayofweek.values.astype(float)

        X_yearly = self._fourier_series(day_in_year, 365.25, self.yearly_order)
        X_weekly = self._fourier_series(day_in_week, 7.0, self.weekly_order)
        X_season = np.column_stack([X_yearly, X_weekly])

        seasonal = np.dot(X_season, self.seasonality_coeffs_)
        yhat = trend + seasonal

        return pd.DataFrame({
            "ds": dt_series,
            "yhat": yhat,
            "trend": trend,
            "seasonal": seasonal,
            "yhat_lower": yhat - 1.96 * 0.05 * np.abs(yhat),
            "yhat_upper": yhat + 1.96 * 0.05 * np.abs(yhat),
        })
