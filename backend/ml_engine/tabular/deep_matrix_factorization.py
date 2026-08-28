"""
ModelForge AI - ML Engine: Deep Matrix Factorization (DMF)
Implements Xue et al. Deep Matrix Factorization Models for Recommender Systems
mapping explicit user rating vectors and item rating vectors into a shared non-linear latent space with Cosine Similarity loss.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class DeepMatrixFactorization:
    """Non-linear deep representation learner for collaborative recommendation."""
    def __init__(self, num_users: int, num_items: int, user_hidden_dims: List[int] = [128, 64], item_hidden_dims: List[int] = [128, 64]):
        self.num_users = num_users
        self.num_items = num_items

        # User branch weights
        self.u_weights = []
        self.u_biases = []
        curr_u = num_items
        for h in user_hidden_dims:
            self.u_weights.append(np.random.normal(0, np.sqrt(2.0 / curr_u), (curr_u, h)))
            self.u_biases.append(np.zeros(h))
            curr_u = h

        # Item branch weights
        self.i_weights = []
        self.i_biases = []
        curr_i = num_users
        for h in item_hidden_dims:
            self.i_weights.append(np.random.normal(0, np.sqrt(2.0 / curr_i), (curr_i, h)))
            self.i_biases.append(np.zeros(h))
            curr_i = h

    def forward(self, user_rating_vector: np.ndarray, item_rating_vector: np.ndarray) -> float:
        # User deep representation
        p = user_rating_vector
        for W, b in zip(self.u_weights, self.u_biases):
            p = np.maximum(0, np.dot(p, W) + b)
        p_norm = p / max(1e-8, np.linalg.norm(p))

        # Item deep representation
        q = item_rating_vector
        for W, b in zip(self.i_weights, self.i_biases):
            q = np.maximum(0, np.dot(q, W) + b)
        q_norm = q / max(1e-8, np.linalg.norm(q))

        # Normalized Cosine similarity
        score = float(np.dot(p_norm, q_norm))
        return max(0.0, score)
