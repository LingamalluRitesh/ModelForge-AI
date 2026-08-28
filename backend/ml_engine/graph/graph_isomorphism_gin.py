"""
ModelForge AI - ML Engine: Graph Isomorphism Network (GIN)
Implements Xu et al. How Powerful are Graph Neural Networks?
achieving maximum expressive power equivalent to the Weisfeiler-Lehman 1-WL graph isomorphism test.
$h_v^{(k)} = 	ext{MLP}^{(k)}\left( (1 + \epsilon^{(k)}) h_v^{(k-1)} + \sum_{u \in \mathcal{N}(v)} h_u^{(k-1)} ight)$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class GINLayer:
    """Expressive multiset graph aggregation layer with learnable epsilon."""
    def __init__(self, in_features: int, out_features: int, eps_trainable: bool = True):
        self.eps = 0.0 if eps_trainable else 0.0

        std = np.sqrt(2.0 / in_features)
        self.W1 = np.random.normal(0, std, (in_features, out_features))
        self.b1 = np.zeros(out_features)
        self.W2 = np.random.normal(0, np.sqrt(2.0 / out_features), (out_features, out_features))
        self.b2 = np.zeros(out_features)

    def forward(self, H: np.ndarray, A: np.ndarray) -> np.ndarray:
        """
        H: (N, in_features)
        A: (N, N) adjacency matrix
        """
        # Sum aggregation over neighbors: A * H
        neighbor_agg = np.dot(A, H)
        # Scaled self-connection: (1 + eps) * H
        combined = (1.0 + self.eps) * H + neighbor_agg

        # 2-layer MLP
        h1 = np.maximum(0, np.dot(combined, self.W1) + self.b1)
        out = np.maximum(0, np.dot(h1, self.W2) + self.b2)
        return out
