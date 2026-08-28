"""
ModelForge AI - ML Engine: Graph Transformer Networks (GTN)
Implements Yun et al. Graph Transformer Networks
learning dynamic meta-path convolutions and attention matrices on heterogeneous graphs via soft adjacency selection.
$\mathbf{A}^{(l+1)} = \sum_{t=1}^{T} \alpha_t^{(l)} \mathbf{A}_t$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class GraphTransformerLayer:
    """Softmax-weighted dynamic meta-path generation layer."""

    def __init__(self, num_edge_types: int):
        self.num_edge_types = num_edge_types
        # Softmax selection weights for generating dynamic meta-paths
        self.w_edge = np.zeros(num_edge_types)

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        shift = z - np.max(z)
        return np.exp(shift) / np.sum(np.exp(shift))

    def forward(self, adj_tensor: np.ndarray) -> np.ndarray:
        """
        adj_tensor: (num_edge_types, N, N)
        returns: (N, N) composite meta-path adjacency matrix
        """
        weights = self._softmax(self.w_edge)
        composite_adj = np.tensordot(weights, adj_tensor, axes=(0, 0))
        return composite_adj


class GraphTransformerNetwork:
    """Stacked GTN with dynamic multi-hop meta-path matrix multiplication and GCN head."""

    def __init__(self, in_features: int, hidden_dim: int, num_classes: int, num_edge_types: int, num_layers: int = 2):
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_edge_types = num_edge_types

        # Multi-layer GTN layers
        self.gtn_layers_1 = [GraphTransformerLayer(num_edge_types) for _ in range(num_layers)]
        self.gtn_layers_2 = [GraphTransformerLayer(num_edge_types) for _ in range(num_layers)]

        # GCN Node Convolution Weights
        std = np.sqrt(2.0 / in_features)
        self.W1 = np.random.normal(0, std, (in_features, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.normal(0, np.sqrt(2.0 / hidden_dim), (hidden_dim, num_classes))
        self.b2 = np.zeros(num_classes)

    def _normalize_adj(self, A: np.ndarray) -> np.ndarray:
        # Add self loops: A_tilde = A + I
        A_tilde = A + np.eye(A.shape[0])
        deg = np.sum(A_tilde, axis=1)
        deg_inv_sqrt = np.power(np.maximum(1e-8, deg), -0.5)
        D_inv_sqrt = np.diag(deg_inv_sqrt)
        return np.dot(np.dot(D_inv_sqrt, A_tilde), D_inv_sqrt)

    def forward(self, X: np.ndarray, adj_tensor: np.ndarray) -> np.ndarray:
        """
        X: (N, in_features)
        adj_tensor: (num_edge_types, N, N)
        """
        # Form composite meta-paths
        H1 = self.gtn_layers_1[0].forward(adj_tensor)
        H2 = self.gtn_layers_2[0].forward(adj_tensor)

        for l in range(1, len(self.gtn_layers_1)):
            H1 = np.dot(H1, self.gtn_layers_1[l].forward(adj_tensor))
            H2 = np.dot(H2, self.gtn_layers_2[l].forward(adj_tensor))

        # Composite multi-hop adjacency
        A_composite = (H1 + H2) * 0.5
        A_norm = self._normalize_adj(A_composite)

        # Graph Convolutional Message Passing
        h1 = np.maximum(0, np.dot(np.dot(A_norm, X), self.W1) + self.b1)
        logits = np.dot(np.dot(A_norm, h1), self.W2) + self.b2
        return logits
