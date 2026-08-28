"""
ModelForge AI - ML Engine: Neural Collaborative Filtering (NCF / NeuMF)
Implements He et al. Neural Collaborative Filtering
fusing Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP) for non-linear user-item interaction scoring.
$\hat{y}_{ui} = \sigma\left( h^T [\phi^{GMF} || \phi^{MLP}] ight)$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class NeuralCollaborativeFiltering:
    """Neural Matrix Factorization combining element-wise GMF dot products and deep MLP."""
    def __init__(self, num_users: int, num_items: int, latent_dim_gmf: int = 32, latent_dim_mlp: int = 32, hidden_dims: List[int] = [64, 32, 16]):
        self.num_users = num_users
        self.num_items = num_items

        # GMF Embeddings
        self.user_embed_gmf = np.random.normal(0, 0.05, (num_users, latent_dim_gmf))
        self.item_embed_gmf = np.random.normal(0, 0.05, (num_items, latent_dim_gmf))

        # MLP Embeddings
        self.user_embed_mlp = np.random.normal(0, 0.05, (num_users, latent_dim_mlp))
        self.item_embed_mlp = np.random.normal(0, 0.05, (num_items, latent_dim_mlp))

        # MLP Layers
        self.mlp_weights = []
        self.mlp_biases = []
        curr_dim = latent_dim_mlp * 2
        for h_dim in hidden_dims:
            self.mlp_weights.append(np.random.normal(0, np.sqrt(2.0 / curr_dim), (curr_dim, h_dim)))
            self.mlp_biases.append(np.zeros(h_dim))
            curr_dim = h_dim

        # NeuMF Final Prediction Vector: (latent_dim_gmf + last_mlp_dim) -> 1
        neu_dim = latent_dim_gmf + hidden_dims[-1]
        self.h_vector = np.random.normal(0, np.sqrt(2.0 / neu_dim), (neu_dim, 1))

    def predict_pair(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        # 1. GMF Branch: Element-wise product
        p_u = self.user_embed_gmf[user_ids]
        q_i = self.item_embed_gmf[item_ids]
        phi_gmf = p_u * q_i  # (B, latent_dim_gmf)

        # 2. MLP Branch: Concatenation
        p_u_mlp = self.user_embed_mlp[user_ids]
        q_i_mlp = self.item_embed_mlp[item_ids]
        phi_mlp = np.concatenate([p_u_mlp, q_i_mlp], axis=-1)
        for W, b in zip(self.mlp_weights, self.mlp_biases):
            phi_mlp = np.maximum(0, np.dot(phi_mlp, W) + b)

        # 3. NeuMF Concatenation & Sigmoid
        fusion = np.concatenate([phi_gmf, phi_mlp], axis=-1)
        logits = np.dot(fusion, self.h_vector)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return probs.reshape(-1)
