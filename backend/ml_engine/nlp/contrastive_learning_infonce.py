"""
ModelForge AI - ML Engine: Contrastive Predictive Coding (InfoNCE)
Implements van den Oord, Li, & Vinyals Representation Learning with Contrastive Predictive Coding
maximizing Mutual Information lower bounds: $I(X; Y) \ge \log(K) - \mathcal{L}_N$.
$\mathcal{L}_N = -\mathbb{E}_{\mathcal{X}} \left[ \log rac{f_k(x_{t+k}, c_t)}{\sum_{x_j \in X} f_k(x_j, c_t)} ight]$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class InfoNCEContrastiveEncoder:
    """Density ratio estimation scoring positive context associations against negative distractors."""
    def __init__(self, feature_dim: int = 128, context_dim: int = 128, temperature: float = 0.07):
        self.dim = feature_dim
        self.tau = temperature
        self.W_k = np.random.normal(0, np.sqrt(2.0 / feature_dim), (context_dim, feature_dim))

    def compute_infonce_loss(self, context_vector: np.ndarray, positive_feature: np.ndarray, negative_features: np.ndarray) -> float:
        # Predict target from context: c_t * W_k
        c_proj = np.dot(context_vector, self.W_k)
        c_norm = c_proj / max(1e-8, np.linalg.norm(c_proj))

        # Positive dot product
        pos_norm = positive_feature / max(1e-8, np.linalg.norm(positive_feature))
        pos_score = np.dot(c_norm, pos_norm) / self.tau

        # Negative dot products
        neg_norms = negative_features / np.maximum(1e-8, np.linalg.norm(negative_features, axis=1, keepdims=True))
        neg_scores = np.dot(neg_norms, c_norm) / self.tau

        all_scores = np.concatenate([[pos_score], neg_scores])
        shift = all_scores - np.max(all_scores)
        probs = np.exp(shift) / np.sum(np.exp(shift))

        loss = -np.log(np.clip(probs[0], 1e-10, 1.0))
        return float(loss)
