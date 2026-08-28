"""
ModelForge AI - ML Engine: Informer ProbSparse Multi-Head Self-Attention
Implements Zhou et al. Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
with ProbSparse Attention $O(L \log L)$ query sampling and Distillation downsampling operations.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class ProbSparseAttention:
    """ProbSparse Self-Attention selecting top-$u$ dominant queries based on Kullback-Leibler measurement."""
    def __init__(self, embed_dim: int = 64, num_heads: int = 4, factor: int = 5):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.factor = factor
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / np.sqrt(self.head_dim)

    def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Q, K, V shapes: (B, num_heads, L, head_dim)
        """
        B, H, L_q, D = Q.shape
        L_k = K.shape[2]

        # Number of top queries to sample: u = factor * ln(L_q)
        u = int(min(L_q, max(1, self.factor * np.ceil(np.log(max(2, L_q))))))

        # 1. Random sample keys to measure query sparsity
        sample_k_idx = np.random.choice(L_k, size=min(L_k, u), replace=False)
        K_sampled = K[:, :, sample_k_idx, :]

        # Dot product scores for sparsity measure
        scores_sample = np.matmul(Q, K_sampled.transpose(0, 1, 3, 2)) * self.scale
        # Query measurement: max(S) - mean(S)
        M = np.max(scores_sample, axis=-1) - np.mean(scores_sample, axis=-1)  # (B, H, L_q)

        # 2. Select top-u active query indices
        top_q_idx = np.argsort(M, axis=-1)[:, :, -u:]

        # Extract top queries
        out = np.zeros_like(Q)
        for b in range(B):
            for h in range(H):
                q_sub = Q[b, h, top_q_idx[b, h], :]  # (u, D)
                scores = np.dot(q_sub, K[b, h].T) * self.scale  # (u, L_k)
                # Softmax
                shift = scores - np.max(scores, axis=-1, keepdims=True)
                attn = np.exp(shift) / np.sum(np.exp(shift), axis=-1, keepdims=True)
                context = np.dot(attn, V[b, h])  # (u, D)
                out[b, h, top_q_idx[b, h], :] = context

        return out
