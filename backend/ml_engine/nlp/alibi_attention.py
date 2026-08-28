"""
ModelForge AI - ML Engine: Attention with Linear Biases (ALiBi)
Implements Press et al. Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation
replacing positional embeddings with static, linearly decaying attention head bias slopes $m$.
$	ext{softmax}\left(\mathbf{q}_i \mathbf{k}_j^T - m \cdot (i - j)ight)$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class ALiBiMultiHeadAttention:
    """Extrapolates to arbitrarily long sequence lengths without retraining."""
    def __init__(self, embed_dim: int = 256, num_heads: int = 8):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / np.sqrt(self.head_dim)

        # Geometric slope sequence m: 2^(-8/n * 1), 2^(-8/n * 2), ...
        ratio = 2.0 ** (-8.0 / num_heads)
        self.slopes = np.array([ratio ** (i + 1) for i in range(num_heads)])

        std = np.sqrt(2.0 / embed_dim)
        self.W_q = np.random.normal(0, std, (embed_dim, embed_dim))
        self.W_k = np.random.normal(0, std, (embed_dim, embed_dim))
        self.W_v = np.random.normal(0, std, (embed_dim, embed_dim))
        self.W_out = np.random.normal(0, std, (embed_dim, embed_dim))

    def _get_alibi_bias(self, seq_len: int) -> np.ndarray:
        """Returns linear distance bias matrix of shape (num_heads, seq_len, seq_len)."""
        positions = np.arange(seq_len)
        dist_matrix = positions[np.newaxis, :] - positions[:, np.newaxis]  # (seq_len, seq_len)
        # Apply head-specific slopes m * (j - i)
        bias = -self.slopes[:, np.newaxis, np.newaxis] * np.abs(dist_matrix)[np.newaxis, :, :]
        return bias

    def forward(self, x: np.ndarray) -> np.ndarray:
        B, S, D = x.shape
        Q = np.dot(x, self.W_q).reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = np.dot(x, self.W_k).reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = np.dot(x, self.W_v).reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) * self.scale
        alibi_bias = self._get_alibi_bias(S)
        scores = scores + alibi_bias[np.newaxis, :, :, :]

        shift = scores - np.max(scores, axis=-1, keepdims=True)
        attn = np.exp(shift) / np.sum(np.exp(shift), axis=-1, keepdims=True)

        context = np.matmul(attn, V).transpose(0, 2, 1, 3).reshape(B, S, D)
        return np.dot(context, self.W_out)
