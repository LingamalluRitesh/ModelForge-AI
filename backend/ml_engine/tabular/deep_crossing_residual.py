"""
ModelForge AI - ML Engine: Deep Crossing Architecture
Implements Shan et al. Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features
with Embedding Layer, Stacking Layer, and Multiple Residual Units (ResNet for Tabular).
$x_{l+1} = x_l + \mathcal{F}(x_l, W_l)$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class ResidualUnit:
    """Tabular Residual Unit with 2 FC layers and Skip Addition."""
    def __init__(self, dim: int):
        self.dim = dim
        std = np.sqrt(2.0 / dim)
        self.W1 = np.random.normal(0, std, (dim, dim))
        self.b1 = np.zeros(dim)
        self.W2 = np.random.normal(0, std, (dim, dim))
        self.b2 = np.zeros(dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h1 = np.maximum(0, np.dot(x, self.W1) + self.b1)
        h2 = np.dot(h1, self.W2) + self.b2
        return np.maximum(0, x + h2)


class DeepCrossingModel:
    """Deep Crossing Deep Tabular Architecture."""
    def __init__(self, num_fields: int, embed_dim: int = 16, num_residual_units: int = 4, num_classes: int = 1):
        self.num_fields = num_fields
        self.embed_dim = embed_dim
        self.total_dim = num_fields * embed_dim

        self.embeddings = np.random.normal(0, 0.1, (num_fields, embed_dim))
        self.res_units = [ResidualUnit(self.total_dim) for _ in range(num_residual_units)]
        self.head_W = np.random.normal(0, np.sqrt(2.0 / self.total_dim), (self.total_dim, num_classes))
        self.head_b = np.zeros(num_classes)

    def forward(self, x: np.ndarray) -> np.ndarray:
        B, M = x.shape
        # Embedding lookup
        emb = x[:, :, np.newaxis] * self.embeddings[np.newaxis, :, :]  # (B, M, D)
        # Stacking layer
        stacked = emb.reshape(B, -1)  # (B, M*D)

        # Residual Units stack
        h = stacked
        for unit in self.res_units:
            h = unit.forward(h)

        # Scoring
        logits = np.dot(h, self.head_W) + self.head_b
        return logits
