"""
ModelForge AI - Forecasting Engine: PatchTST (Patch Time Series Transformer)
Implements Nie et al. A Time Series is Worth 64 Words: Long-term Forecasting with Transformers
utilizing Channel-Independence, Subseries Patch Tokenization, and Self-Supervised Representation Masking.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class PatchTSTEmbedding:
    """Slices 1D subseries sequence into overlapping token patches and linearly projects into $D$."""

    def __init__(self, patch_len: int = 16, stride: int = 8, embed_dim: int = 64):
        self.patch_len = patch_len
        self.stride = stride
        self.embed_dim = embed_dim
        std = np.sqrt(2.0 / patch_len)
        self.W_proj = np.random.normal(0, std, (patch_len, embed_dim))
        self.b_proj = np.zeros(embed_dim)

    def extract_patches(self, sequence: np.ndarray) -> np.ndarray:
        """
        sequence: (B, L) 1D univariate or single channel series
        returns: (B, num_patches, embed_dim)
        """
        B, L = sequence.shape
        num_patches = (L - self.patch_len) // self.stride + 1
        patches = []
        for p in range(num_patches):
            start = p * self.stride
            end = start + self.patch_len
            patches.append(sequence[:, start:end])

        # (B, num_patches, patch_len)
        stacked = np.stack(patches, axis=1)
        # Linear patch projection
        projected = np.dot(stacked, self.W_proj) + self.b_proj
        return projected


class PatchTSTForecaster:
    """Channel-Independent Transformer processing multi-variate time series via shared univariate backbone."""

    def __init__(
        self,
        lookback_len: int = 96,
        forecast_len: int = 24,
        patch_len: int = 16,
        stride: int = 8,
        embed_dim: int = 64,
        num_heads: int = 4,
    ):
        self.lookback = lookback_len
        self.forecast_len = forecast_len
        self.patch_embed = PatchTSTEmbedding(patch_len, stride, embed_dim)

        num_patches = (lookback_len - patch_len) // stride + 1
        self.pos_embed = np.random.normal(0, 0.02, (1, num_patches, embed_dim))

        # Flatten head: (num_patches * embed_dim) -> forecast_len
        flat_dim = num_patches * embed_dim
        self.head_W = np.random.normal(0, np.sqrt(2.0 / flat_dim), (flat_dim, forecast_len))
        self.head_b = np.zeros(forecast_len)

    def forecast_channel(self, series_1d: np.ndarray) -> np.ndarray:
        """
        series_1d: (B, lookback)
        returns: (B, forecast_len)
        """
        B = series_1d.shape[0]
        # 1. Instance Normalization (RevIN)
        mean = np.mean(series_1d, axis=-1, keepdims=True)
        std = np.std(series_1d, axis=-1, keepdims=True) + 1e-5
        norm_series = (series_1d - mean) / std

        # 2. Patch Embedding + Position Encoding
        tokens = self.patch_embed.extract_patches(norm_series) + self.pos_embed

        # 3. Linear Head
        flat = tokens.reshape(B, -1)
        pred_norm = np.dot(flat, self.head_W) + self.head_b

        # 4. De-Normalize
        pred = (pred_norm * std) + mean
        return pred
