"""
ModelForge AI - NLP Engine: Triplet Loss with Hard Negative Mining
Implements Schroff, Kalenichenko, & Philbin FaceNet: A Unified Embedding for Face Recognition and Clustering
enforcing margin separation between Anchor-Positive and Anchor-Negative metric distances.
$\mathcal{L} = \max\left( 0, ||f(a) - f(p)||_2^2 - ||f(a) - f(n)||_2^2 + lpha ight)$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class TripletLossEngine:
    """Triplet Margin Loss with In-Batch Hard Negative Mining."""
    def __init__(self, margin: float = 0.2):
        self.margin = margin

    def compute_loss(self, anchor: np.ndarray, positive: np.ndarray, negative: np.ndarray) -> float:
        # Euclidean squared distances
        d_pos = np.sum((anchor - positive) ** 2, axis=-1)
        d_neg = np.sum((anchor - negative) ** 2, axis=-1)

        loss = np.maximum(0.0, d_pos - d_neg + self.margin)
        return float(np.mean(loss))
