"""
ModelForge AI - ML Engine: DLinear & NLinear Forecasting Models
Implements Zeng et al. Are Transformers Effective for Time Series?
with Simple Moving Average Trend-Seasonal Decomposition and Single-Layer Linear Forecasters.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class MovingAverageBlock:
    """Decomposes series into Moving Average Trend and Residual Seasonal components."""
    def __init__(self, kernel_size: int = 25):
        self.kernel_size = kernel_size

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # x shape: (B, L)
        pad = (self.kernel_size - 1) // 2
        x_pad = np.pad(x, ((0, 0), (pad, pad)), mode="edge")
        trend = np.convolve(x_pad[0], np.ones(self.kernel_size) / self.kernel_size, mode="valid").reshape(1, -1)
        seasonal = x - trend
        return seasonal, trend


class DLinearModel:
    """DLinear Model with independent Trend and Seasonal linear projections."""
    def __init__(self, seq_len: int = 96, pred_len: int = 24, ma_kernel: int = 25):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.decomposer = MovingAverageBlock(ma_kernel)

        self.W_trend = np.random.normal(0, 1.0 / seq_len, (seq_len, pred_len))
        self.b_trend = np.zeros(pred_len)
        self.W_seasonal = np.random.normal(0, 1.0 / seq_len, (seq_len, pred_len))
        self.b_seasonal = np.zeros(pred_len)

    def predict(self, history: np.ndarray) -> np.ndarray:
        x = np.asarray(history, dtype=float).reshape(1, -1)
        seasonal, trend = self.decomposer.forward(x)

        trend_pred = np.dot(trend, self.W_trend) + self.b_trend
        seasonal_pred = np.dot(seasonal, self.W_seasonal) + self.b_seasonal

        forecast = trend_pred + seasonal_pred
        return forecast[0]
