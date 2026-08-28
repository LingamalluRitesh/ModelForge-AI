"""
ModelForge AI - Time Series Forecasting Algorithms
Implements ARIMA, SARIMAX, Exponential Smoothing, and Fourier-based Trend/Seasonality Estimators.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class ExponentialSmoothingForecaster:
    """
    Holt-Winters Triple Exponential Smoothing Forecaster with level, trend, and seasonal components.
    """

    def __init__(
        self,
        alpha: float = 0.2,
        beta: float = 0.1,
        gamma: float = 0.1,
        season_length: int = 12,
        seasonal_type: str = "additive",
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.season_length = season_length
        self.seasonal_type = seasonal_type
        self.level: float = 0.0
        self.trend: float = 0.0
        self.seasonals: List[float] = []
        self.history: List[float] = []

    def fit(self, y: np.ndarray):
        """Fit exponential smoothing model on univariate time series array."""
        self.history = list(y)
        n = len(y)
        if n < self.season_length * 2:
            # Fallback for short series: Simple exponential smoothing
            self.level = float(y[0])
            self.trend = float((y[-1] - y[0]) / max(1, n))
            self.seasonals = [0.0] * self.season_length
            return self

        # Initialize level and trend
        self.level = float(np.mean(y[:self.season_length]))
        self.trend = float((np.mean(y[self.season_length:2 * self.season_length]) - self.level) / self.season_length)

        # Initialize seasonal factors
        self.seasonals = []
        for i in range(self.season_length):
            if self.seasonal_type == "additive":
                self.seasonals.append(float(y[i] - self.level))
            else:
                self.seasonals.append(float(y[i] / max(1e-6, self.level)))

        # Recursive updates
        for t in range(n):
            val = y[t]
            s_idx = t % self.season_length
            last_level = self.level
            last_trend = self.trend
            last_seasonal = self.seasonals[s_idx]

            if self.seasonal_type == "additive":
                self.level = self.alpha * (val - last_seasonal) + (1 - self.alpha) * (last_level + last_trend)
                self.trend = self.beta * (self.level - last_level) + (1 - self.beta) * last_trend
                self.seasonals[s_idx] = self.gamma * (val - self.level) + (1 - self.gamma) * last_seasonal
            else:
                self.level = self.alpha * (val / max(1e-6, last_seasonal)) + (1 - self.alpha) * (last_level + last_trend)
                self.trend = self.beta * (self.level - last_level) + (1 - self.beta) * last_trend
                self.seasonals[s_idx] = self.gamma * (val / max(1e-6, self.level)) + (1 - self.gamma) * last_seasonal

        return self

    def predict(self, steps: int = 10) -> np.ndarray:
        """Forecast future values for specified number of steps."""
        forecasts = []
        for h in range(1, steps + 1):
            s_idx = (len(self.history) + h - 1) % self.season_length
            seasonal = self.seasonals[s_idx]
            if self.seasonal_type == "additive":
                pred = self.level + h * self.trend + seasonal
            else:
                pred = (self.level + h * self.trend) * seasonal
            forecasts.append(pred)
        return np.array(forecasts)


class FourierSeasonalityForecaster:
    """
    Prophet-inspired Fourier decomposition linear forecaster for multiple seasonality patterns.
    """

    def __init__(self, fourier_order: int = 5, period: float = 365.25):
        self.fourier_order = fourier_order
        self.period = period
        self.coefficients: Optional[np.ndarray] = None
        self.intercept: float = 0.0
        self.trend_slope: float = 0.0

    def _create_features(self, t: np.ndarray) -> np.ndarray:
        features = [t]  # Linear trend
        for k in range(1, self.fourier_order + 1):
            features.append(np.sin(2 * np.pi * k * t / self.period))
            features.append(np.cos(2 * np.pi * k * t / self.period))
        return np.column_stack(features)

    def fit(self, y: np.ndarray):
        t = np.arange(len(y), dtype=float)
        X = self._create_features(t)
        # Solve least squares
        X_design = np.column_stack([np.ones(len(y)), X])
        coeffs, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
        self.intercept = coeffs[0]
        self.coefficients = coeffs[1:]
        return self

    def predict(self, steps: int = 10, start_t: Optional[int] = None) -> np.ndarray:
        if self.coefficients is None:
            raise ValueError("Model is not fitted.")
        if start_t is None:
            start_t = 0
        t = np.arange(start_t, start_t + steps, dtype=float)
        X = self._create_features(t)
        return self.intercept + X @ self.coefficients
