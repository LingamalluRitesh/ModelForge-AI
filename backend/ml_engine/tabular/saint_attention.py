"""
ModelForge AI - ML Engine: SAINT (Self-Attention and Intersample Attention Transformer)
Implements Somepalli et al. SAINT: Improved Neural Networks for Tabular Data via Row and Column Attention
with Self-Attention over Features (Columns) and Intersample Attention over Data Instances (Rows).
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class FeatureSelfAttentionBlock:
    """Self-Attention over Feature Dimension (Columns)."""
    def __init__(self, embed_dim: int = 64, num_heads: int = 4):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / np.sqrt(self.head_dim)

        std = np.sqrt(2.0 / embed_dim)
        self.W_q = np.random.normal(0, std, (embed_dim, embed_dim))
        self.W_k = np.random.normal(0, std, (embed_dim, embed_dim))
        self.W_v = np.random.normal(0, std, (embed_dim, embed_dim))
        self.W_out = np.random.normal(0, std, (embed_dim, embed_dim))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: (B, num_features, embed_dim)
        """
        B, M, D = x.shape
        Q = np.dot(x, self.W_q).reshape(B, M, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = np.dot(x, self.W_k).reshape(B, M, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = np.dot(x, self.W_v).reshape(B, M, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) * self.scale
        shift = scores - np.max(scores, axis=-1, keepdims=True)
        attn = np.exp(shift) / np.sum(np.exp(shift), axis=-1, keepdims=True)

        context = np.matmul(attn, V).transpose(0, 2, 1, 3).reshape(B, M, D)
        out = np.dot(context, self.W_out)
        return x + out


class IntersampleAttentionBlock:
    """Intersample Attention over Sample Rows (Instances)."""
    def __init__(self, embed_dim: int = 64, num_heads: int = 4):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / np.sqrt(self.head_dim)

        std = np.sqrt(2.0 / embed_dim)
        self.W_q = np.random.normal(0, std, (embed_dim, embed_dim))
        self.W_k = np.random.normal(0, std, (embed_dim, embed_dim))
        self.W_v = np.random.normal(0, std, (embed_dim, embed_dim))
        self.W_out = np.random.normal(0, std, (embed_dim, embed_dim))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: (B, num_features, embed_dim) -> transpose to (num_features, B, embed_dim)
        """
        B, M, D = x.shape
        x_t = x.transpose(1, 0, 2)  # (M, B, D)

        Q = np.dot(x_t, self.W_q).reshape(M, B, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = np.dot(x_t, self.W_k).reshape(M, B, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = np.dot(x_t, self.W_v).reshape(M, B, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) * self.scale
        shift = scores - np.max(scores, axis=-1, keepdims=True)
        attn = np.exp(shift) / np.sum(np.exp(shift), axis=-1, keepdims=True)

        context = np.matmul(attn, V).transpose(0, 2, 1, 3).reshape(M, B, D)
        out_t = np.dot(context, self.W_out)
        out = out_t.transpose(1, 0, 2)  # (B, M, D)
        return x + out


class SAINTModel:
    """Complete SAINT Tabular Architecture."""
    def __init__(self, num_features: int, embed_dim: int = 64, num_layers: int = 2, num_classes: int = 2):
        self.num_features = num_features
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # Embeddings
        self.W_emb = np.random.normal(0, 0.1, (num_features, embed_dim))
        self.b_emb = np.zeros((num_features, embed_dim))

        self.self_attns = [FeatureSelfAttentionBlock(embed_dim) for _ in range(num_layers)]
        self.inter_attns = [IntersampleAttentionBlock(embed_dim) for _ in range(num_layers)]

        self.head_W = np.random.normal(0, 0.05, (num_features * embed_dim, num_classes))
        self.head_b = np.zeros(num_classes)

    def forward(self, x: np.ndarray) -> np.ndarray:
        B, M = x.shape
        # Embed
        tokens = x[:, :, np.newaxis] * self.W_emb[np.newaxis, :, :] + self.b_emb[np.newaxis, :, :]

        # Alternating Column and Row Attention
        h = tokens
        for s_attn, i_attn in zip(self.self_attns, self.inter_attns):
            h = s_attn.forward(h)
            h = i_attn.forward(h)

        flat = h.reshape(B, -1)
        logits = np.dot(flat, self.head_W) + self.head_b
        return logits
