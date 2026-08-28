"""
ModelForge AI - ML Engine: Attentional Factorization Machines (AFM)
Implements Xiao et al. Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks
using attention-based pooling over 2nd-order element-wise Hadamard products.
$\hat{y}_{AFM}(x) = w_0 + \sum_{i=1}^m w_i x_i + \mathbf{p}^T \sum_{i=1}^m \sum_{j=i+1}^m a_{ij} (v_i \odot v_j) x_i x_j$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class AttentionalFactorizationMachine:
    """AFM with attention-weighted feature interaction pooling."""
    def __init__(self, num_fields: int, embed_dim: int = 16, attention_dim: int = 16):
        self.num_fields = num_fields
        self.embed_dim = embed_dim
        self.num_pairs = num_fields * (num_fields - 1) // 2

        self.w_linear = np.random.normal(0, 0.05, (num_fields, 1))
        self.b_linear = 0.0
        self.V = np.random.normal(0, np.sqrt(2.0 / embed_dim), (num_fields, embed_dim))

        # Attention network weights
        self.W_att = np.random.normal(0, np.sqrt(2.0 / embed_dim), (embed_dim, attention_dim))
        self.b_att = np.zeros(attention_dim)
        self.h_att = np.random.normal(0, np.sqrt(2.0 / attention_dim), (attention_dim, 1))
        self.p_vec = np.random.normal(0, np.sqrt(2.0 / embed_dim), (embed_dim, 1))

    def forward(self, x: np.ndarray) -> np.ndarray:
        B, M = x.shape
        linear_part = np.dot(x, self.w_linear) + self.b_linear

        # Element-wise product pair matrix
        xv = x[:, :, np.newaxis] * self.V[np.newaxis, :, :]  # (B, M, D)
        pair_products = []
        for i in range(M):
            for j in range(i + 1, M):
                pair_products.append(xv[:, i, :] * xv[:, j, :])
        pairs_tensor = np.stack(pair_products, axis=1)  # (B, num_pairs, D)

        # Attention score computation: h^T relu(W * (v_i (x) v_j) + b)
        att_hidden = np.maximum(0, np.dot(pairs_tensor, self.W_att) + self.b_att)  # (B, num_pairs, att_dim)
        att_scores = np.dot(att_hidden, self.h_att)                                # (B, num_pairs, 1)

        # Softmax over pairs
        shift = att_scores - np.max(att_scores, axis=1, keepdims=True)
        att_weights = np.exp(shift) / np.sum(np.exp(shift), axis=1, keepdims=True)

        # Attention-weighted pooling
        pooled = np.sum(att_weights * pairs_tensor, axis=1)  # (B, D)
        afm_part = np.dot(pooled, self.p_vec)                # (B, 1)

        return linear_part + afm_part
