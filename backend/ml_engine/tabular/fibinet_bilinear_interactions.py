"""
ModelForge AI - ML Engine: FiBiNET (Feature Importance and Bilinear feature Interaction Network)
Implements Huang et al. FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction
with Squeeze-and-Excitation (SENET) field gating and Bilinear Interaction (Bi-Interaction) tensors.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class SENETFieldGating:
    """Squeeze-and-Excitation block dynamically reweighting tabular field importance."""

    def __init__(self, num_fields: int, reduction_ratio: int = 2):
        self.num_fields = num_fields
        r_dim = max(1, num_fields // reduction_ratio)
        std = np.sqrt(2.0 / num_fields)
        self.W1 = np.random.normal(0, std, (num_fields, r_dim))
        self.b1 = np.zeros(r_dim)
        self.W2 = np.random.normal(0, np.sqrt(2.0 / r_dim), (r_dim, num_fields))
        self.b2 = np.zeros(num_fields)

    def forward(self, field_embeddings: np.ndarray) -> np.ndarray:
        """
        field_embeddings: (B, num_fields, embed_dim)
        """
        # 1. Squeeze: global average pooling across embed_dim
        z = np.mean(field_embeddings, axis=-1)  # (B, num_fields)

        # 2. Excitation: two-layer MLP + Sigmoid
        h = np.maximum(0, np.dot(z, self.W1) + self.b1)
        a = 1.0 / (1.0 + np.exp(- (np.dot(h, self.W2) + self.b2)))  # (B, num_fields)

        # 3. Re-weight field embeddings
        a_expanded = a[..., np.newaxis]
        return field_embeddings * a_expanded


class BilinearInteractionLayer:
    """Computes Bilinear Interactions: $p = v_i \cdot W \odot v_j$ across all $(i, j)$ field pairs."""

    def __init__(self, num_fields: int, embed_dim: int):
        self.num_fields = num_fields
        self.embed_dim = embed_dim
        self.num_pairs = (num_fields * (num_fields - 1)) // 2
        # Field-All bilinear weight matrix W (D, D)
        std = np.sqrt(2.0 / embed_dim)
        self.W = np.random.normal(0, std, (embed_dim, embed_dim))

    def forward(self, field_embeddings: np.ndarray) -> np.ndarray:
        """
        field_embeddings: (B, num_fields, embed_dim)
        returns: (B, num_pairs * embed_dim)
        """
        B, M, D = field_embeddings.shape
        # Project all fields: v_i * W
        projected = np.dot(field_embeddings, self.W)  # (B, M, D)

        pair_interactions = []
        for i in range(M):
            for j in range(i + 1, M):
                interaction = projected[:, i, :] * field_embeddings[:, j, :]
                pair_interactions.append(interaction)

        return np.concatenate(pair_interactions, axis=-1)
