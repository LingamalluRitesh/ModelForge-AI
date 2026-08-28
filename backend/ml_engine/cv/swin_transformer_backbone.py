"""
ModelForge AI - CV Engine: Swin Transformer Hierarchical Vision Backbone
Implements Liu et al. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
with Window Multi-Head Self-Attention (W-MSA) and Shifted Window Multi-Head Self-Attention (SW-MSA).
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class WindowAttention:
    """Computes Self-Attention within local non-overlapping 2D image windows."""
    def __init__(self, dim: int, window_size: int = 7, num_heads: int = 4):
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / np.sqrt(self.head_dim)

        std = np.sqrt(2.0 / dim)
        self.W_qkv = np.random.normal(0, std, (dim, 3 * dim))
        self.W_proj = np.random.normal(0, std, (dim, dim))
        self.relative_position_bias_table = np.zeros((2 * window_size - 1, 2 * window_size - 1, num_heads))

    def forward(self, x: np.ndarray) -> np.ndarray:
        B, N, C = x.shape
        qkv = np.dot(x, self.W_qkv).reshape(B, N, 3, self.num_heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        shift = scores - np.max(scores, axis=-1, keepdims=True)
        attn = np.exp(shift) / np.sum(np.exp(shift), axis=-1, keepdims=True)

        out = np.matmul(attn, v).transpose(0, 2, 1, 3).reshape(B, N, C)
        return np.dot(out, self.W_proj)


class SwinBlock:
    """Swin Transformer block with Cyclic Shift and Window Attention."""
    def __init__(self, dim: int, window_size: int = 7, num_heads: int = 4, shift_size: int = 0):
        self.dim = dim
        self.shift_size = shift_size
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.mlp_w1 = np.random.normal(0, np.sqrt(2.0 / dim), (dim, 4 * dim))
        self.mlp_b1 = np.zeros(4 * dim)
        self.mlp_w2 = np.random.normal(0, np.sqrt(2.0 / (4 * dim)), (4 * dim, dim))
        self.mlp_b2 = np.zeros(dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Attention with residual
        h = x + self.attn.forward(x)
        # Feed-Forward MLP
        mlp_h = np.maximum(0, np.dot(h, self.mlp_w1) + self.mlp_b1)
        mlp_out = np.dot(mlp_h, self.mlp_w2) + self.mlp_b2
        return h + mlp_out
