"""
ModelForge AI - ML Engine: TabNet Attentive Transformer & Sparse Prior Masking
Implements Arik & Pfister TabNet: Attentive Interpretable Tabular Learning
with Prior Scale Masking $P[i] = \prod_{j=1}^{i-1} (\gamma - M[j])$, Feature Selection Masks, and Sparsemax Attention.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class AttentiveTransformer:
    """Computes sparse feature selection mask $M[i]$ conditioned on prior attention scale $P[i-1]$."""
    def __init__(self, in_features: int, num_fields: int, gamma: float = 1.3):
        self.in_features = in_features
        self.num_fields = num_fields
        self.gamma = gamma

        std = np.sqrt(2.0 / in_features)
        self.W_attn = np.random.normal(0, std, (in_features, num_fields))
        self.b_attn = np.zeros(num_fields)

    def _sparsemax(self, z: np.ndarray) -> np.ndarray:
        B, D = z.shape
        z_sorted = np.sort(z, axis=-1)[:, ::-1]
        z_cumsum = np.cumsum(z_sorted, axis=-1)
        k_indices = np.arange(1, D + 1)
        tau_condition = 1.0 + k_indices * z_sorted > z_cumsum
        k_z = np.sum(tau_condition, axis=-1, keepdims=True)
        tau_z = (np.take_along_axis(z_cumsum, k_z - 1, axis=-1) - 1.0) / k_z
        return np.maximum(0.0, z - tau_z)

    def forward(self, processed_features: np.ndarray, prior_scale: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        processed_features: (B, in_features)
        prior_scale: (B, num_fields)
        returns: (feature_mask, updated_prior_scale)
        """
        # Linear projection
        h = np.dot(processed_features, self.W_attn) + self.b_attn
        # Modulate by prior scale
        h_scaled = h * prior_scale

        # Sparsemax feature mask M[i]
        mask = self._sparsemax(h_scaled)

        # Update prior scale: P[i] = P[i-1] * (gamma - M[i])
        updated_prior = prior_scale * (self.gamma - mask)
        return mask, updated_prior
