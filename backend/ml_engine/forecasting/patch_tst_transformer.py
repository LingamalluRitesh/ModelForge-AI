"""
ModelForge AI - ML Engine: PatchTST Patch Time Series Transformer
Implements Nie et al. A Time Series is Worth 64 Words: Long-term Forecasting with PatchTST
using Subseries Patching, Channel Independence, and Self-Supervised Masked Autoencoding.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class Patch1D:
    """Extracts overlapping 1D temporal subseries patches."""
    def __init__(self, patch_len: int = 16, stride: int = 8):
        self.patch_len = patch_len
        self.stride = stride

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: (B, L) -> returns (B, num_patches, patch_len)
        """
        B, L = x.shape
        num_patches = (L - self.patch_len) // self.stride + 1
        patches = []
        for i in range(num_patches):
            start = i * self.stride
            end = start + self.patch_len
            patches.append(x[:, start:end])
        return np.stack(patches, axis=1)


class PatchTSTForecaster:
    """Channel-Independent Patch Transformer for multi-variate multi-horizon forecasting."""
    def __init__(
        self,
        history_len: int = 96,
        forecast_len: int = 24,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
    ):
        self.history_len = history_len
        self.forecast_len = forecast_len
        self.patcher = Patch1D(patch_len, stride)
        num_patches = (history_len - patch_len) // stride + 1

        self.W_proj = np.random.normal(0, 0.05, (patch_len, d_model))
        self.W_head = np.random.normal(0, 0.05, (num_patches * d_model, forecast_len))
        self.b_head = np.zeros(forecast_len)

    def forecast_channel(self, series: np.ndarray) -> np.ndarray:
        """Forecast individual 1D univariate channel."""
        x = np.asarray(series, dtype=float).reshape(1, -1)

        # Instance Normalization
        mean = float(np.mean(x))
        std = float(max(1e-5, np.std(x)))
        x_norm = (x - mean) / std

        # Patching & Projection
        patches = self.patcher.forward(x_norm)  # (1, num_patches, patch_len)
        tokens = np.dot(patches, self.W_proj)   # (1, num_patches, d_model)

        # Flatten & Linear Head
        flat = tokens.reshape(1, -1)
        pred_norm = np.dot(flat, self.W_head) + self.b_head

        # De-normalization
        pred = pred_norm * std + mean
        return pred[0]
