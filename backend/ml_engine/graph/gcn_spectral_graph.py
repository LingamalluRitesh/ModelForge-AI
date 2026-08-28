"""
ModelForge AI - ML Engine: Graph Convolutional Networks (GCN)
Implements Kipf & Welling Semi-Supervised Classification with Graph Convolutional Networks
using Renormalization Trick $\hat{A} = 	ilde{D}^{-1/2} 	ilde{A} 	ilde{D}^{-1/2}$ and Layer Propagation.
$H^{(l+1)} = \sigma\left(	ilde{D}^{-rac{1}{2}} 	ilde{A} 	ilde{D}^{-rac{1}{2}} H^{(l)} W^{(l)}ight)$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class GCNLayer:
    """First-order localized spectral graph convolution layer."""
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features

        std = np.sqrt(2.0 / in_features)
        self.W = np.random.normal(0, std, (in_features, out_features))
        self.bias = np.zeros(out_features)

    def _normalize_adjacency(self, A: np.ndarray) -> np.ndarray:
        # Renormalization trick: A_tilde = A + I
        N = A.shape[0]
        A_tilde = A + np.eye(N)
        deg = np.sum(A_tilde, axis=1)
        deg_inv_sqrt = np.power(np.maximum(1e-10, deg), -0.5)
        D_inv_sqrt = np.diag(deg_inv_sqrt)
        return np.dot(D_inv_sqrt, np.dot(A_tilde, D_inv_sqrt))

    def forward(self, H: np.ndarray, A: np.ndarray) -> np.ndarray:
        """
        H: (N, in_features) node features
        A: (N, N) graph adjacency matrix
        """
        A_norm = self._normalize_adjacency(A)
        # Message passing: A_norm * H * W
        out = np.dot(A_norm, np.dot(H, self.W)) + self.bias
        return np.maximum(0, out)
