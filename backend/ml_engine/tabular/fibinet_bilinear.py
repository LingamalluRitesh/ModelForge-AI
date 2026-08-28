"""
ModelForge AI - ML Engine: FiBiNET Bilinear Feature Interactions
Implements Huang et al. FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction
with Squeeze-and-Excitation (SENET) field weighting and Bilinear Interaction layers (All-Interaction and Each-Interaction).
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class SENetLayer:
    """Squeeze-and-Excitation field-level dynamic weight assignment."""
    def __init__(self, num_fields: int, reduction_ratio: int = 2):
        self.num_fields = num_fields
        r_dim = max(1, num_fields // reduction_ratio)

        self.W1 = np.random.normal(0, 0.1, (num_fields, r_dim))
        self.b1 = np.zeros(r_dim)
        self.W2 = np.random.normal(0, 0.1, (r_dim, num_fields))
        self.b2 = np.zeros(num_fields)

    def forward(self, field_embeddings: np.ndarray) -> np.ndarray:
        # Squeeze: Global pooling per field: (B, M)
        squeezed = np.mean(field_embeddings, axis=-1)
        # Excitation: MLP -> Sigmoid
        h = np.maximum(0, np.dot(squeezed, self.W1) + self.b1)
        weights = 1.0 / (1.0 + np.exp(-(np.dot(h, self.W2) + self.b2)))  # (B, M)
        # Scale field embeddings
        return field_embeddings * weights[:, :, np.newaxis]


class BilinearInteractionLayer:
    """Computes Hadamard / Inner-Product Bilinear interactions: $p_{ij} = v_i W_{ij} v_j$."""
    def __init__(self, num_fields: int, embed_dim: int):
        self.num_fields = num_fields
        self.embed_dim = embed_dim
        # Shared bilinear interaction weight matrix: (embed_dim, embed_dim)
        self.W = np.random.normal(0, np.sqrt(2.0 / embed_dim), (embed_dim, embed_dim))

    def forward(self, embeddings: np.ndarray) -> np.ndarray:
        B, M, D = embeddings.shape
        # Vectorized bilinear pairs
        interactions = []
        for i in range(M):
            for j in range(i + 1, M):
                # v_i * W * v_j
                vi_w = np.dot(embeddings[:, i, :], self.W)
                p_ij = vi_w * embeddings[:, j, :]
                interactions.append(p_ij)

        return np.concatenate(interactions, axis=-1)
