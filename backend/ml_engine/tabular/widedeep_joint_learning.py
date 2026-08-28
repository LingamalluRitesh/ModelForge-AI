"""
ModelForge AI - ML Engine: Wide & Deep Joint Learning
Implements Cheng et al. Wide & Deep Learning for Recommender Systems
jointly training linear models (for memorization) and deep neural networks (for generalization).
$P(Y = 1 | \mathbf{x}) = \sigma\left( \mathbf{w}_{wide}^T [\mathbf{x}, \phi(\mathbf{x})] + \mathbf{w}_{deep}^T a^{(L)} + b ight)$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class WideAndDeepModel:
    """Wide & Deep joint learning architecture with cross-product transformations."""
    def __init__(self, num_wide_features: int, num_deep_features: int, embed_dim: int = 16, hidden_dims: List[int] = [128, 64], num_classes: int = 1):
        self.num_wide = num_wide_features
        self.num_deep = num_deep_features
        self.embed_dim = embed_dim

        # Wide branch linear weights
        self.w_wide = np.random.normal(0, 0.05, (num_wide_features, 1))

        # Deep branch embeddings and DNN
        self.embeddings = np.random.normal(0, 0.1, (num_deep_features, embed_dim))
        self.deep_weights = []
        self.deep_biases = []
        curr_dim = num_deep_features * embed_dim
        for h_dim in hidden_dims:
            self.deep_weights.append(np.random.normal(0, np.sqrt(2.0 / curr_dim), (curr_dim, h_dim)))
            self.deep_biases.append(np.zeros(h_dim))
            curr_dim = h_dim

        self.w_deep = np.random.normal(0, np.sqrt(2.0 / curr_dim), (curr_dim, 1))
        self.bias = 0.0

    def forward(self, x_wide: np.ndarray, x_deep: np.ndarray) -> np.ndarray:
        B = x_wide.shape[0]

        # 1. Wide Memorization Part: w_wide^T x_wide
        wide_out = np.dot(x_wide, self.w_wide)  # (B, 1)

        # 2. Deep Generalization Part: w_deep^T a^(L)
        deep_emb = x_deep[:, :, np.newaxis] * self.embeddings[np.newaxis, :, :]  # (B, M, D)
        flat_emb = deep_emb.reshape(B, -1)
        h = flat_emb
        for W, b in zip(self.deep_weights, self.deep_biases):
            h = np.maximum(0, np.dot(h, W) + b)
        deep_out = np.dot(h, self.w_deep)  # (B, 1)

        logits = wide_out + deep_out + self.bias
        return logits
