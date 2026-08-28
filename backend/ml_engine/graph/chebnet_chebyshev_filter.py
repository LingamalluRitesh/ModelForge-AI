"""
ModelForge AI - ML Engine: Chebyshev Spectral Graph Convolution (ChebNet)
Implements Defferrard, Bresson, & Vandergheynst Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering
using truncated Chebyshev polynomial expansions $g_	heta \star x pprox \sum_{k=0}^K 	heta_k T_k(	ilde{L}) x$.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class ChebNetConvolution:
    """K-th order localized Chebyshev spectral graph convolution."""
    def __init__(self, in_features: int, out_features: int, K: int = 3):
        self.in_features = in_features
        self.out_features = out_features
        self.K = K

        # Learnable filter weights for each order k in [0, K-1]
        std = np.sqrt(2.0 / (K * in_features))
        self.weights = [np.random.normal(0, std, (in_features, out_features)) for _ in range(K)]
        self.bias = np.zeros(out_features)

    def _scaled_laplacian(self, A: np.ndarray) -> np.ndarray:
        """Scaled Laplacian: $	ilde{L} = rac{2}{\lambda_{max}} L - I_N$."""
        N = A.shape[0]
        deg = np.sum(A, axis=1)
        deg_inv_sqrt = np.power(np.maximum(1e-10, deg), -0.5)
        L = np.eye(N) - np.dot(np.diag(deg_inv_sqrt), np.dot(A, np.diag(deg_inv_sqrt)))
        # Lambda max is bounded by 2
        lambda_max = 2.0
        return (2.0 / lambda_max) * L - np.eye(N)

    def forward(self, H: np.ndarray, A: np.ndarray) -> np.ndarray:
        """
        Recursive Chebyshev polynomial computation:
        $T_0(L) = I$, $T_1(L) = L$, $T_k(L) = 2 L T_{k-1}(L) - T_{k-2}(L)$
        """
        L_tilde = self._scaled_laplacian(A)

        # T0 = H
        T0 = H
        out = np.dot(T0, self.weights[0])

        if self.K > 1:
            # T1 = L_tilde * H
            T1 = np.dot(L_tilde, H)
            out += np.dot(T1, self.weights[1])

            # Recurrence for k >= 2
            T_prev2 = T0
            T_prev1 = T1
            for k in range(2, self.K):
                T_curr = 2.0 * np.dot(L_tilde, T_prev1) - T_prev2
                out += np.dot(T_curr, self.weights[k])
                T_prev2 = T_prev1
                T_prev1 = T_curr

        out += self.bias
        return np.maximum(0, out)
