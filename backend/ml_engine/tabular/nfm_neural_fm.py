"""
ModelForge AI - ML Engine: Neural Factorization Machines (NFM)
Implements He & Chua Neural Factorization Machines for Sparse Predictive Analytics
with Bi-Interaction Pooling layer: $f_{BI}(\mathcal{V}_x) = \sum_{i=1}^n \sum_{j=i+1}^n x_i v_i \odot x_j v_j$.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class BiInteractionPooling:
    """Computes second-order element-wise vector interaction in linear $O(k D)$ time."""
    def forward(self, embedding_matrix: np.ndarray) -> np.ndarray:
        # embedding_matrix shape: (B, M, D)
        sum_embeddings = np.sum(embedding_matrix, axis=1)          # (B, D)
        sum_sq_embeddings = sum_embeddings ** 2                   # (B, D)

        sq_embeddings = embedding_matrix ** 2                     # (B, M, D)
        sq_sum_embeddings = np.sum(sq_embeddings, axis=1)         # (B, D)

        # $1/2 [ (\sum v)^2 - \sum v^2 ]$
        return 0.5 * (sum_sq_embeddings - sq_sum_embeddings)


class NFMModel:
    """Neural Factorization Machine Architecture."""
    def __init__(self, num_fields: int, embed_dim: int = 16, hidden_dims: List[int] = [128, 64], num_classes: int = 1):
        self.num_fields = num_fields
        self.embed_dim = embed_dim
        self.bi_pooling = BiInteractionPooling()

        self.w_linear = np.random.normal(0, 0.05, (num_fields, 1))
        self.b_linear = 0.0
        self.V = np.random.normal(0, np.sqrt(2.0 / embed_dim), (num_fields, embed_dim))

        self.deep_weights = []
        self.deep_biases = []
        curr_dim = embed_dim
        for h_dim in hidden_dims:
            self.deep_weights.append(np.random.normal(0, np.sqrt(2.0 / curr_dim), (curr_dim, h_dim)))
            self.deep_biases.append(np.zeros(h_dim))
            curr_dim = h_dim

        self.head_W = np.random.normal(0, np.sqrt(2.0 / curr_dim), (curr_dim, num_classes))
        self.head_b = np.zeros(num_classes)

    def forward(self, x: np.ndarray) -> np.ndarray:
        B = x.shape[0]
        # Linear component
        linear_part = np.dot(x, self.w_linear) + self.b_linear

        # Bi-Interaction Pooling
        xv = x[:, :, np.newaxis] * self.V[np.newaxis, :, :]  # (B, M, D)
        pooled = self.bi_pooling.forward(xv)                 # (B, D)

        # Deep MLP
        h = pooled
        for W, b in zip(self.deep_weights, self.deep_biases):
            h = np.maximum(0, np.dot(h, W) + b)

        deep_part = np.dot(h, self.head_W) + self.head_b
        return linear_part + deep_part
