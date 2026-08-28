"""
ModelForge AI - ML Engine: AutoInt Self-Attentive Tabular Interactions
Implements Song et al. AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks
with Multi-Head Explicit Feature Interaction mapping in high-order polynomial vector spaces.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class AutoIntInteractionLayer:
    """Multi-head explicit feature interaction layer."""
    def __init__(self, num_fields: int, embed_dim: int, num_heads: int = 4):
        self.num_fields = num_fields
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / np.sqrt(self.head_dim)

        std = np.sqrt(2.0 / embed_dim)
        self.W_q = np.random.normal(0, std, (embed_dim, embed_dim))
        self.W_k = np.random.normal(0, std, (embed_dim, embed_dim))
        self.W_v = np.random.normal(0, std, (embed_dim, embed_dim))
        self.W_res = np.random.normal(0, std, (embed_dim, embed_dim))

    def forward(self, field_embeddings: np.ndarray) -> np.ndarray:
        """
        field_embeddings: (B, num_fields, embed_dim)
        """
        B, M, D = field_embeddings.shape

        # Linear projections
        Q = np.dot(field_embeddings, self.W_q).reshape(B, M, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = np.dot(field_embeddings, self.W_k).reshape(B, M, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = np.dot(field_embeddings, self.W_v).reshape(B, M, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Scaled attention scores: (B, num_heads, M, M)
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) * self.scale
        shift = scores - np.max(scores, axis=-1, keepdims=True)
        attn_weights = np.exp(shift) / np.sum(np.exp(shift), axis=-1, keepdims=True)

        # Context output: (B, num_heads, M, head_dim) -> (B, M, D)
        context = np.matmul(attn_weights, V).transpose(0, 2, 1, 3).reshape(B, M, D)

        # Residual skip connection
        residual = np.dot(field_embeddings, self.W_res)
        return np.maximum(0, context + residual)
