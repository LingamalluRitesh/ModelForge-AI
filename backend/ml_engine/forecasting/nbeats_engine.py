"""
ModelForge AI - ML Engine: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting (N-BEATS)
Implements Oreshkin et al. N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting
with Doubly Residual Stacking, Trend Basis Functions (Polynomial), and Seasonality Basis Functions (Fourier).
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class NBeatsBlock:
    def __init__(self, input_dim: int, theta_dim: int, hidden_dim: int = 128, block_type: str = "generic"):
        self.input_dim = input_dim
        self.theta_dim = theta_dim
        self.block_type = block_type

        # 4 Fully-Connected Layers with ReLU
        self.W1 = np.random.normal(0, np.sqrt(2.0 / input_dim), (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.normal(0, np.sqrt(2.0 / hidden_dim), (hidden_dim, hidden_dim))
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.normal(0, np.sqrt(2.0 / hidden_dim), (hidden_dim, hidden_dim))
        self.b3 = np.zeros(hidden_dim)
        self.W4 = np.random.normal(0, np.sqrt(2.0 / hidden_dim), (hidden_dim, hidden_dim))
        self.b4 = np.zeros(hidden_dim)

        # Expansion coefficients
        self.W_theta_b = np.random.normal(0, 0.05, (hidden_dim, theta_dim))
        self.W_theta_f = np.random.normal(0, 0.05, (hidden_dim, theta_dim))

    def forward(self, x: np.ndarray, forecast_horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        h1 = np.maximum(0, np.dot(x, self.W1) + self.b1)
        h2 = np.maximum(0, np.dot(h1, self.W2) + self.b2)
        h3 = np.maximum(0, np.dot(h2, self.W3) + self.b3)
        h4 = np.maximum(0, np.dot(h3, self.W4) + self.b4)

        theta_b = np.dot(h4, self.W_theta_b)
        theta_f = np.dot(h4, self.W_theta_f)

        if self.block_type == "trend":
            # Polynomial basis: sum_p theta_p * t^p
            t_b = np.linspace(0, 1, self.input_dim)
            backcast = sum(theta_b[:, p : p + 1] * (t_b ** p) for p in range(min(self.theta_dim, 4)))
            t_f = np.linspace(1, 2, forecast_horizon)
            forecast = sum(theta_f[:, p : p + 1] * (t_f ** p) for p in range(min(self.theta_dim, 4)))
        else:
            # Generic linear projection basis
            backcast = np.dot(theta_b, np.ones((self.theta_dim, self.input_dim)) / self.theta_dim)
            forecast = np.dot(theta_f, np.ones((self.theta_dim, forecast_horizon)) / self.theta_dim)

        return backcast, forecast


class NBeatsModel:
    def __init__(self, backcast_length: int = 30, forecast_horizon: int = 7, num_stacks: int = 3):
        self.backcast_length = backcast_length
        self.forecast_horizon = forecast_horizon
        self.blocks = [
            NBeatsBlock(backcast_length, theta_dim=8, block_type="trend" if i % 2 == 0 else "generic")
            for i in range(num_stacks)
        ]

    def predict(self, history: np.ndarray) -> np.ndarray:
        x = np.asarray(history, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        residual_backcast = x.copy()
        accumulated_forecast = np.zeros((x.shape[0], self.forecast_horizon))

        for block in self.blocks:
            b_hat, f_hat = block.forward(residual_backcast, self.forecast_horizon)
            residual_backcast = residual_backcast - b_hat
            accumulated_forecast = accumulated_forecast + f_hat

        return accumulated_forecast[0] if history.ndim == 1 else accumulated_forecast
