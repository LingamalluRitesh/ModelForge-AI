"""
ModelForge AI - ML Engine: Temporal Graph Attention Networks (TGAT)
Implements Xu et al. Inductive Representation Learning on Temporal Graphs with Time-Aware Self-Attention
using continuous-time functional kernel mapping $\Phi(t) = \left[\cos(\omega_1 t), \sin(\omega_1 t), \dotsight]$.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class ContinuousTimeKernel:
    """Bochner Theorem Continuous Time Fourier Embedding."""
    def __init__(self, time_dim: int = 64):
        self.time_dim = time_dim
        # Log-spaced frequency basis
        self.omega = 1.0 / (10000.0 ** (np.arange(0, time_dim, 2).astype(float) / time_dim))

    def transform(self, delta_t: np.ndarray) -> np.ndarray:
        # delta_t shape: (B,) -> output (B, time_dim)
        angles = np.outer(delta_t, self.omega)
        return np.concatenate([np.cos(angles), np.sin(angles)], axis=-1) / np.sqrt(self.time_dim)


class TemporalGraphAttentionLayer:
    """Time-aware self-attention over dynamic historical edge events."""
    def __init__(self, node_dim: int, edge_dim: int, time_dim: int = 64, num_heads: int = 4):
        self.time_encoder = ContinuousTimeKernel(time_dim)
        in_dim = node_dim + edge_dim + time_dim

        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads

        std = np.sqrt(2.0 / in_dim)
        self.W_q = np.random.normal(0, std, (node_dim + time_dim, node_dim))
        self.W_k = np.random.normal(0, std, (in_dim, node_dim))
        self.W_v = np.random.normal(0, std, (in_dim, node_dim))

    def forward(self, node_feat: np.ndarray, neighbor_feats: np.ndarray, edge_feats: np.ndarray, timestamps: np.ndarray, curr_time: float) -> np.ndarray:
        delta_t = np.maximum(0.0, curr_time - timestamps)
        time_emb = self.time_encoder.transform(delta_t)

        # Concatenate features
        k_in = np.concatenate([neighbor_feats, edge_feats, time_emb], axis=-1)
        v_in = k_in

        q_time = self.time_encoder.transform(np.array([0.0]))[0]
        q_in = np.concatenate([node_feat, q_time])

        Q = np.dot(q_in, self.W_q)
        K = np.dot(k_in, self.W_k)
        V = np.dot(v_in, self.W_v)

        # Attention scores
        scores = np.dot(K, Q) / np.sqrt(self.head_dim)
        shift = scores - np.max(scores)
        attn = np.exp(shift) / np.sum(np.exp(shift))

        context = np.dot(attn, V)
        return np.maximum(0, node_feat + context)
