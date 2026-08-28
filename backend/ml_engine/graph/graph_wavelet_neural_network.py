"""
ModelForge AI - ML Engine: Graph Wavelet Neural Network (GWNN)
Implements Xu et al. Graph Wavelet Neural Network replacing traditional graph Fourier transforms
with localized, sparse, multi-resolution Graph Wavelet Transforms $\psi_s = U e^{-s \Lambda} U^T$.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class GraphWaveletTransform:
    """Computes localized spectral graph wavelet transform matrices."""
    def __init__(self, scale_s: float = 1.0):
        self.s = scale_s
        self.psi_: Optional[np.ndarray] = None
        self.psi_inv_: Optional[np.ndarray] = None

    def fit(self, adjacency: np.ndarray):
        N = adjacency.shape[0]
        deg = np.diag(np.sum(adjacency, axis=1))
        laplacian = deg - adjacency

        # Eigendecomposition: L = U Lambda U^T
        eigvals, U = np.linalg.eigh(laplacian)

        # Graph Wavelet kernel: psi_s = U exp(-s * Lambda) U^T
        heat_kernel = np.exp(-self.s * eigvals)
        self.psi_ = np.dot(U, np.dot(np.diag(heat_kernel), U.T))
        # Inverse wavelet transform
        self.psi_inv_ = np.linalg.pinv(self.psi_)
        return self

    def transform(self, node_features: np.ndarray) -> np.ndarray:
        """Forward Graph Wavelet Transform: $\hat{X} = \psi_s^{-1} X$."""
        return np.dot(self.psi_inv_, node_features)

    def inverse_transform(self, wavelet_features: np.ndarray) -> np.ndarray:
        """Inverse Graph Wavelet Transform: $X = \psi_s \hat{X}$."""
        return np.dot(self.psi_, wavelet_features)
