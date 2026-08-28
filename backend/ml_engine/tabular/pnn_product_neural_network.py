"""
ModelForge AI - ML Engine: Product-based Neural Network (PNN)
Implements Qu et al. Product-based Neural Networks for User Response Prediction
with Inner Product Layer (IPNN) and Outer Product Layer (OPNN) for non-linear feature interaction learning.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class InnerProductLayer:
    """Computes pair-wise inner products: $p = [\langle v_i, v_j angle]_{i < j}$."""
    def __init__(self, num_fields: int, embed_dim: int):
        self.num_fields = num_fields
        self.embed_dim = embed_dim
        self.num_pairs = num_fields * (num_fields - 1) // 2

    def forward(self, embeddings: np.ndarray) -> np.ndarray:
        B, M, D = embeddings.shape
        inner_products = []
        for i in range(M):
            for j in range(i + 1, M):
                dot_prod = np.sum(embeddings[:, i, :] * embeddings[:, j, :], axis=-1, keepdims=True)
                inner_products.append(dot_prod)
        return np.concatenate(inner_products, axis=-1)


class OuterProductLayer:
    """Computes pair-wise matrix outer product feature crosses."""
    def __init__(self, num_fields: int, embed_dim: int, kernel_dim: int = 16):
        self.num_fields = num_fields
        self.embed_dim = embed_dim
        self.kernel_dim = kernel_dim
        self.W_op = np.random.normal(0, 0.05, (embed_dim, kernel_dim, embed_dim))

    def forward(self, embeddings: np.ndarray) -> np.ndarray:
        B, M, D = embeddings.shape
        # Sum of embeddings across fields
        f_sum = np.sum(embeddings, axis=1)  # (B, D)
        # Outer product: f_sum (x) f_sum contracted with kernel
        outer_prod = np.einsum("bi,ikj,bj->bk", f_sum, self.W_op, f_sum)
        return outer_prod


class PNNModel:
    """Product-based Neural Network combining linear signals and non-linear product signals."""
    def __init__(self, num_fields: int, embed_dim: int = 16, hidden_dims: List[int] = [128, 64], num_classes: int = 1):
        self.num_fields = num_fields
        self.embed_dim = embed_dim
        self.ipnn = InnerProductLayer(num_fields, embed_dim)
        self.opnn = OuterProductLayer(num_fields, embed_dim)

        linear_dim = num_fields * embed_dim
        product_dim = self.ipnn.num_pairs + 16
        in_dim = linear_dim + product_dim

        self.weights = []
        self.biases = []
        curr_dim = in_dim
        for h_dim in hidden_dims:
            self.weights.append(np.random.normal(0, np.sqrt(2.0 / curr_dim), (curr_dim, h_dim)))
            self.biases.append(np.zeros(h_dim))
            curr_dim = h_dim

        self.head_W = np.random.normal(0, np.sqrt(2.0 / curr_dim), (curr_dim, num_classes))
        self.head_b = np.zeros(num_classes)

    def forward(self, x_embeddings: np.ndarray) -> np.ndarray:
        B = x_embeddings.shape[0]
        linear_signals = x_embeddings.reshape(B, -1)
        ip_signals = self.ipnn.forward(x_embeddings)
        op_signals = self.opnn.forward(x_embeddings)

        combined = np.concatenate([linear_signals, ip_signals, op_signals], axis=-1)
        h = combined
        for W, b in zip(self.weights, self.biases):
            h = np.maximum(0, np.dot(h, W) + b)
        logits = np.dot(h, self.head_W) + self.head_b
        return logits
