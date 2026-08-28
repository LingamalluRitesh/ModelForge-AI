"""
ModelForge AI - NLP Engine: Rotary Position Embedding (RoPE)
Implements Su et al. RoFormer: Enhanced Transformer with Rotary Position Embedding
incorporating relative position information through dynamic orthogonal 2D rotation matrices.
$\mathbf{R}_{\Theta, m}^d = 	ext{diag}\left( \mathbf{R}_{	heta_1, m}, \mathbf{R}_{	heta_2, m}, \dots, \mathbf{R}_{	heta_{d/2}, m} ight)$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class RotaryPositionEmbedding:
    """Applies Rotary 2D coordinate rotations to Query and Key tensors."""
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        self.dim = dim
        self.max_seq_len = max_seq_len
        # Frequency theta_i = base^(-2(i-1)/dim)
        theta = 1.0 / (base ** (np.arange(0, dim, 2).astype(float) / dim))

        # Position indices [0, 1, ..., max_seq_len - 1]
        seq_idx = np.arange(max_seq_len)
        # Outer product: shape (max_seq_len, dim // 2)
        idx_theta = np.outer(seq_idx, theta)

        self.cos_cached = np.cos(idx_theta)
        self.sin_cached = np.sin(idx_theta)

    def _rotate_half(self, x: np.ndarray) -> np.ndarray:
        """[-x2, x1] rotation helper."""
        d = x.shape[-1] // 2
        x1 = x[..., :d]
        x2 = x[..., d:]
        return np.concatenate([-x2, x1], axis=-1)

    def apply_rope(self, x: np.ndarray) -> np.ndarray:
        """
        x: (B, num_heads, seq_len, head_dim)
        """
        seq_len = x.shape[2]
        cos = np.repeat(self.cos_cached[:seq_len], 2, axis=-1)[np.newaxis, np.newaxis, :, :]
        sin = np.repeat(self.sin_cached[:seq_len], 2, axis=-1)[np.newaxis, np.newaxis, :, :]

        return (x * cos) + (self._rotate_half(x) * sin)
