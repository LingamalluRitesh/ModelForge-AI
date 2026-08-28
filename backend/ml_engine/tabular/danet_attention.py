"""
ModelForge AI - ML Engine: Deep Abstract Networks for Tabular Data (DANet)
Implements Chen et al. DANet: Deep Abstract Networks for Tabular Data Classification and Regression
with Abstract Layer feature selection and Attention-based Feature Rebuilding.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class AbstractBlock:
    """Extracts high-level feature abstracts and rebuilds raw feature space."""
    def __init__(self, in_features: int, num_abstracts: int = 16):
        self.in_dim = in_features
        self.k = num_abstracts

        std = np.sqrt(2.0 / in_features)
        self.W_select = np.random.normal(0, std, (in_features, num_abstracts))
        self.b_select = np.zeros(num_abstracts)
        self.W_rebuild = np.random.normal(0, np.sqrt(2.0 / num_abstracts), (num_abstracts, in_features))
        self.b_rebuild = np.zeros(in_features)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # 1. Feature Abstract Extraction
        abstracts = np.maximum(0, np.dot(x, self.W_select) + self.b_select)
        # 2. Feature Rebuilding
        rebuilt = np.maximum(0, np.dot(abstracts, self.W_rebuild) + self.b_rebuild)
        return abstracts, rebuilt


class DANetModel:
    """Stacked DANet Architecture with Shortcut Residual Connections."""
    def __init__(self, in_features: int, num_blocks: int = 3, num_classes: int = 1):
        self.blocks = [AbstractBlock(in_features) for _ in range(num_blocks)]
        self.head_W = np.random.normal(0, np.sqrt(2.0 / (num_blocks * 16)), (num_blocks * 16, num_classes))
        self.head_b = np.zeros(num_classes)

    def forward(self, x: np.ndarray) -> np.ndarray:
        curr = x
        all_abstracts = []

        for block in self.blocks:
            abstracts, rebuilt = block.forward(curr)
            all_abstracts.append(abstracts)
            curr = curr + rebuilt  # Shortcut residual

        stacked_abstracts = np.concatenate(all_abstracts, axis=-1)
        logits = np.dot(stacked_abstracts, self.head_W) + self.head_b
        return logits
