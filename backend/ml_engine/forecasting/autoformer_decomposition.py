"""
ModelForge AI - ML Engine: Autoformer Auto-Correlation Transformer
Implements Wu et al. Autoformer: Decomposition Transformers with Auto-Correlation
replacing point-wise self-attention with series-level subseries Auto-Correlation and progressive trend decomposition.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class AutoCorrelationMechanism:
    """Series-level Auto-Correlation computing phase similarity via Fast Fourier Transform."""
    def __init__(self, d_model: int = 64, factor: int = 3):
        self.d_model = d_model
        self.factor = factor
        self.W_q = np.random.normal(0, 0.05, (d_model, d_model))
        self.W_k = np.random.normal(0, 0.05, (d_model, d_model))
        self.W_v = np.random.normal(0, 0.05, (d_model, d_model))
        self.W_out = np.random.normal(0, 0.05, (d_model, d_model))

    def forward(self, queries: np.ndarray, keys: np.ndarray, values: np.ndarray) -> np.ndarray:
        # queries, keys, values shapes: (B, L, D)
        B, L, D = queries.shape
        Q = np.dot(queries, self.W_q)
        K = np.dot(keys, self.W_k)
        V = np.dot(values, self.W_v)

        # FFT based Wiener-Khinchin auto-correlation
        q_fft = np.fft.rfft(Q, axis=1)
        k_fft = np.fft.rfft(K, axis=1)
        corr_fft = q_fft * np.conj(k_fft)
        corr = np.fft.irfft(corr_fft, n=L, axis=1)  # (B, L, D)

        # Softmax over correlation lags
        shift = corr - np.max(corr, axis=1, keepdims=True)
        weights = np.exp(shift) / np.sum(np.exp(shift), axis=1, keepdims=True)

        context = weights * V
        return np.dot(context, self.W_out)
