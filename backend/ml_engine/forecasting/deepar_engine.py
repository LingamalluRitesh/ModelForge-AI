"""
ModelForge AI - ML Engine: DeepAR Probabilistic Autoregressive Forecasting
Implements Salinas et al. DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks
modeling Gaussian predictive distributions with Monte Carlo sample rollout trajectories.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class DeepARForecaster:
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, num_samples: int = 100):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_samples = num_samples

        # Recurrent cell weights
        self.W_x = np.random.normal(0, 0.1, (input_dim, hidden_dim))
        self.W_h = np.random.normal(0, 0.1, (hidden_dim, hidden_dim))
        self.bias = np.zeros(hidden_dim)

        # Distribution parameter projection heads (Mean and Variance)
        self.W_mu = np.random.normal(0, 0.1, (hidden_dim, 1))
        self.b_mu = np.zeros(1)
        self.W_sigma = np.random.normal(0, 0.1, (hidden_dim, 1))
        self.b_sigma = np.zeros(1)

    def sample_forecast(self, history: np.ndarray, forecast_steps: int = 10) -> Dict[str, np.ndarray]:
        h = np.zeros(self.hidden_dim)
        for y in history:
            inp = np.array([y])
            h = np.tanh(np.dot(inp, self.W_x) + np.dot(h, self.W_h) + self.bias)

        # Monte Carlo rollout trajectories: (num_samples, forecast_steps)
        trajectories = np.zeros((self.num_samples, forecast_steps))

        for s in range(self.num_samples):
            h_step = h.copy()
            y_curr = history[-1]
            for t in range(forecast_steps):
                inp = np.array([y_curr])
                h_step = np.tanh(np.dot(inp, self.W_x) + np.dot(h_step, self.W_h) + self.bias)

                mu = float(np.dot(h_step, self.W_mu) + self.b_mu)
                sigma = float(np.log1p(np.exp(np.dot(h_step, self.W_sigma) + self.b_sigma))) + 1e-3

                # Sample from normal distribution
                y_sample = np.random.normal(mu, sigma)
                trajectories[s, t] = y_sample
                y_curr = y_sample

        # Compute probabilistic quantile intervals
        p10 = np.quantile(trajectories, 0.10, axis=0)
        p50 = np.quantile(trajectories, 0.50, axis=0)
        p90 = np.quantile(trajectories, 0.90, axis=0)

        return {
            "mean": np.mean(trajectories, axis=0),
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "samples": trajectories,
        }
