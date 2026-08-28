"""
ModelForge AI - ML Engine: Simple Contrastive Learning of Sentence Embeddings (SimCSE)
Implements Gao, Yao, & Chen SimCSE: Simple Contrastive Learning of Sentence Embeddings
with Dropout as Data Augmentation and InfoNCE Contrastive Loss.
$\ell_i = -\log rac{e^{	ext{sim}(h_i, h_i^+) / 	au}}{\sum_{j=1}^N e^{	ext{sim}(h_i, h_j^+) / 	au}}$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class SimCSESentenceEncoder:
    """Unsupervised Contrastive Sentence Embedding Engine."""
    def __init__(self, embed_dim: int = 256, temperature: float = 0.05, dropout_rate: float = 0.1):
        self.embed_dim = embed_dim
        self.tau = temperature
        self.dropout_rate = dropout_rate

        std = np.sqrt(2.0 / embed_dim)
        self.W_proj = np.random.normal(0, std, (embed_dim, embed_dim))
        self.b_proj = np.zeros(embed_dim)

    def _apply_dropout(self, x: np.ndarray) -> np.ndarray:
        mask = np.random.binomial(1, 1.0 - self.dropout_rate, size=x.shape) / (1.0 - self.dropout_rate)
        return x * mask

    def encode_pair(self, text_vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Pass identical sentence representations through two independent dropout masks."""
        # View 1
        z1 = np.dot(self._apply_dropout(text_vectors), self.W_proj) + self.b_proj
        z1 = z1 / np.maximum(1e-8, np.linalg.norm(z1, axis=-1, keepdims=True))

        # View 2 (Positive pair)
        z2 = np.dot(self._apply_dropout(text_vectors), self.W_proj) + self.b_proj
        z2 = z2 / np.maximum(1e-8, np.linalg.norm(z2, axis=-1, keepdims=True))

        return z1, z2

    def info_nce_loss(self, z1: np.ndarray, z2: np.ndarray) -> float:
        """Compute InfoNCE contrastive loss treating in-batch samples as negative examples."""
        N = z1.shape[0]
        # Cosine similarity matrix: (N, N)
        sim_matrix = np.dot(z1, z2.T) / self.tau

        # Exponentiate
        shift_sim = sim_matrix - np.max(sim_matrix, axis=1, keepdims=True)
        exp_sim = np.exp(shift_sim)

        # Numerator: diagonal positive pair elements
        pos_sim = np.diag(exp_sim)
        denom = np.sum(exp_sim, axis=1)

        loss = -np.mean(np.log(np.clip(pos_sim / np.maximum(1e-10, denom), 1e-10, 1.0)))
        return float(loss)
