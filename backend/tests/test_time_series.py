"""
ModelForge AI - Time Series Forecaster Unit Tests
"""

import numpy as np
import pytest
from ml_engine.algorithms.time_series import ExponentialSmoothingForecaster, FourierSeasonalityForecaster


def test_exponential_smoothing():
    np.random.seed(42)
    t = np.arange(60)
    # Trend + seasonal pattern
    y = 10 + 0.5 * t + 5 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 0.5, 60)

    model = ExponentialSmoothingForecaster(alpha=0.3, beta=0.1, gamma=0.1, season_length=12)
    model.fit(y)
    preds = model.predict(steps=6)

    assert len(preds) == 6
    assert preds[0] > y[0]  # Positive upward trend


def test_fourier_seasonality():
    np.random.seed(42)
    t = np.arange(100)
    y = 50 + 0.2 * t + 8 * np.sin(2 * np.pi * t / 30) + np.random.normal(0, 0.5, 100)

    model = FourierSeasonalityForecaster(fourier_order=3, period=30.0)
    model.fit(y)
    preds = model.predict(steps=10, start_t=100)

    assert len(preds) == 10
    assert not np.isnan(preds).any()
