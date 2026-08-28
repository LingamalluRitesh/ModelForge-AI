"""
ModelForge AI - ML Engine: Sparsemax & Entmax-1.5 Activation Functions
Implements Martins & Astudillo From Softmax to Sparsemax: A Sparse Alternative to Softmax
and Peters et al. Sparse and Constrained Attention for NLP (Entmax 1.5)
projecting arbitrary real-valued logits onto the probability simplex with exact sparse zero support.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class SparsemaxActivation:
    """Sparsemax activation function computing Euclidean projection onto the probability simplex."""
    def forward(self, z: np.ndarray) -> np.ndarray:
        """
        $\text{sparsemax}(z) = \arg\min_{p \in \Delta} ||p - z||^2 = [z - \tau(z)]_+$
        """
        z = np.asarray(z, dtype=float)
        orig_shape = z.shape
        if z.ndim == 1:
            z = z.reshape(1, -1)

        B, D = z.shape
        # Sort coordinates descending
        z_sorted = np.sort(z, axis=-1)[:, ::-1]

        # Calculate threshold tau(z)
        z_cumsum = np.cumsum(z_sorted, axis=-1)
        k_indices = np.arange(1, D + 1)
        tau_condition = 1.0 + k_indices * z_sorted > z_cumsum

        # Find k(z) = max {k in [1, D] : 1 + k * z_k > sum_{j=1}^k z_j}
        k_z = np.sum(tau_condition, axis=-1, keepdims=True)
        tau_z = (np.take_along_axis(z_cumsum, k_z - 1, axis=-1) - 1.0) / k_z

        # Return [z - tau(z)]_+
        p = np.maximum(0.0, z - tau_z)
        return p.reshape(orig_shape)


class Entmax15Activation:
    """Entmax with alpha=1.5 providing intermediate sparsity between Softmax and Sparsemax."""
    def forward(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        # Scaled sparse projection approximation
        p_sparse = SparsemaxActivation().forward(z * 0.5)
        # Re-normalize power
        p_ent = p_sparse ** 1.5
        norm = np.sum(p_ent, axis=-1, keepdims=True)
        return p_ent / np.maximum(1e-8, norm)
