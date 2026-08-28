"""
ModelForge AI - ML Engine: Deep & Cross Network v2 (DCN-v2)
Implements Wang et al. DCN V2: Improved Deep & Cross Network for Feature Crosses in Web-scale Learning
with explicit vector and matrix cross layers computing $x_{l+1} = x_0 \odot (W_l x_l + b_l) + x_l$.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class CrossLayer:
    """Explicit polynomial feature crossing layer."""
    def __init__(self, in_features: int):
        self.in_features = in_features
        std = np.sqrt(2.0 / in_features)
        self.W = np.random.normal(0, std, (in_features, in_features))
        self.b = np.zeros(in_features)

    def forward(self, x0: np.ndarray, xl: np.ndarray) -> np.ndarray:
        """
        $x_{l+1} = x_0 \odot (x_l W_l + b_l) + x_l$
        """
        crossed = x0 * (np.dot(xl, self.W) + self.b)
        return crossed + xl


class DCNv2Model:
    """Stacked / Parallel Deep & Cross Network."""
    def __init__(self, in_features: int, num_cross_layers: int = 3, deep_hidden_dims: List[int] = [128, 64], num_classes: int = 1):
        self.cross_layers = [CrossLayer(in_features) for _ in range(num_cross_layers)]

        # Deep branch
        self.deep_weights = []
        self.deep_biases = []
        curr_dim = in_features
        for h_dim in deep_hidden_dims:
            self.deep_weights.append(np.random.normal(0, np.sqrt(2.0 / curr_dim), (curr_dim, h_dim)))
            self.deep_biases.append(np.zeros(h_dim))
            curr_dim = h_dim

        # Final combination head: (in_features + last_deep_dim) -> num_classes
        comb_dim = in_features + deep_hidden_dims[-1]
        self.head_W = np.random.normal(0, np.sqrt(2.0 / comb_dim), (comb_dim, num_classes))
        self.head_b = np.zeros(num_classes)

    def forward(self, x0: np.ndarray) -> np.ndarray:
        # Cross network branch
        xl = x0
        for cross in self.cross_layers:
            xl = cross.forward(x0, xl)

        # Deep network branch
        h = x0
        for W, b in zip(self.deep_weights, self.deep_biases):
            h = np.maximum(0, np.dot(h, W) + b)

        # Concatenate cross and deep branches
        combined = np.concatenate([xl, h], axis=-1)
        logits = np.dot(combined, self.head_W) + self.head_b
        return logits
