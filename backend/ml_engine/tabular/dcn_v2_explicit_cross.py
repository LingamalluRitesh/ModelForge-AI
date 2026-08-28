"""
ModelForge AI - ML Engine: Deep & Cross Network (DCN-v2) Architecture
Implements Wang et al. DCN V2: Improved Deep & Cross Network for Feature Crosses in Large-Scale Recommender Systems
with explicit matrix-vector polynomial tensor crosses: $x_{l+1} = x_0 \odot (W_l x_l + b_l) + x_l$.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class CrossNetworkV2Layer:
    """Explicit matrix cross layer computing polynomial degree interaction tensors."""

    def __init__(self, in_features: int):
        self.in_dim = in_features
        std = np.sqrt(2.0 / in_features)
        self.W = np.random.normal(0, std, (in_features, in_features))
        self.b = np.zeros(in_features)

    def forward(self, x0: np.ndarray, xl: np.ndarray) -> np.ndarray:
        """
        x0: input feature tensor (B, D)
        xl: l-th layer feature tensor (B, D)
        """
        linear = np.dot(xl, self.W) + self.b
        cross = x0 * linear
        return cross + xl


class DeepAndCrossNetworkV2:
    """Combines stacked Cross Network layers with parallel Deep MLP network."""

    def __init__(
        self,
        in_features: int,
        num_cross_layers: int = 3,
        deep_hidden_dims: List[int] = [128, 64],
        num_classes: int = 1,
    ):
        self.in_dim = in_features
        self.cross_layers = [CrossNetworkV2Layer(in_features) for _ in range(num_cross_layers)]

        # Deep MLP Branch
        self.deep_weights = []
        self.deep_biases = []
        curr = in_features
        for h in deep_hidden_dims:
            self.deep_weights.append(np.random.normal(0, np.sqrt(2.0 / curr), (curr, h)))
            self.deep_biases.append(np.zeros(h))
            curr = h

        # Output head: (in_features + last_deep_dim) -> num_classes
        total_dim = in_features + deep_hidden_dims[-1]
        self.head_W = np.random.normal(0, np.sqrt(2.0 / total_dim), (total_dim, num_classes))
        self.head_b = np.zeros(num_classes)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Cross Branch
        x_cross = x
        for layer in self.cross_layers:
            x_cross = layer.forward(x, x_cross)

        # Deep Branch
        x_deep = x
        for W, b in zip(self.deep_weights, self.deep_biases):
            x_deep = np.maximum(0, np.dot(x_deep, W) + b)

        # Stacking Combination
        fusion = np.concatenate([x_cross, x_deep], axis=-1)
        logits = np.dot(fusion, self.head_W) + self.head_b
        return logits
