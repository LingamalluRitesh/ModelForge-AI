"""
ModelForge AI - ML Engine: Multi-Query Attention (MQA) & Grouped-Query Attention (GQA)
Implements Shazeer Fast Transformer Decoding (MQA) and Ainslie et al. GQA: Training Generalized Multi-Query Transformer Models
sharing Key-Value heads across Query head groups to reduce KV-cache memory bandwidth by 8x.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class GroupedQueryAttention:
    """Grouped-Query Attention (GQA) with $H_Q$ query heads and $H_{KV}$ key-value heads."""
    def __init__(self, embed_dim: int = 256, num_query_heads: int = 8, num_kv_heads: int = 2):
        self.embed_dim = embed_dim
        self.num_q_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_query_heads // num_kv_heads
        self.head_dim = embed_dim // num_query_heads
        self.scale = 1.0 / np.sqrt(self.head_dim)

        std = np.sqrt(2.0 / embed_dim)
        self.W_q = np.random.normal(0, std, (embed_dim, embed_dim))
        self.W_k = np.random.normal(0, std, (embed_dim, num_kv_heads * self.head_dim))
        self.W_v = np.random.normal(0, std, (embed_dim, num_kv_heads * self.head_dim))
        self.W_out = np.random.normal(0, std, (embed_dim, embed_dim))

    def forward(self, x: np.ndarray) -> np.ndarray:
        B, S, D = x.shape
        Q = np.dot(x, self.W_q).reshape(B, S, self.num_q_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = np.dot(x, self.W_k).reshape(B, S, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = np.dot(x, self.W_v).reshape(B, S, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Broadcast/Repeat KV heads to match query group size
        K_rep = np.repeat(K, self.group_size, axis=1)  # (B, num_q_heads, S, head_dim)
        V_rep = np.repeat(V, self.group_size, axis=1)  # (B, num_q_heads, S, head_dim)

        # Attention
        scores = np.matmul(Q, K_rep.transpose(0, 1, 3, 2)) * self.scale
        shift = scores - np.max(scores, axis=-1, keepdims=True)
        attn = np.exp(shift) / np.sum(np.exp(shift), axis=-1, keepdims=True)

        context = np.matmul(attn, V_rep).transpose(0, 2, 1, 3).reshape(B, S, D)
        return np.dot(context, self.W_out)
