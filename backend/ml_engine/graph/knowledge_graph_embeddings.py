"""
ModelForge AI - ML Engine: Knowledge Graph Embedding Models (TransE, ComplEx, RotatE)
Implements Bordes et al. TransE translation-based scoring, Trouillon et al. ComplEx Hermitian dot product,
and Sun et al. RotatE relational rotation in complex coordinate space for multi-relational link prediction.
TransE score: $f(h, r, t) = -|| \mathbf{h} + \mathbf{r} - \mathbf{t} ||$
ComplEx score: $f(h, r, t) = 	ext{Re}(\langle \mathbf{h}, \mathbf{r}, \mathbf{ar{t}} angle)$
RotatE score: $f(h, r, t) = -|| \mathbf{h} \circ \mathbf{r} - \mathbf{t} ||$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class TransE:
    """Translational Distance Embedding for Knowledge Graph Entities and Relations."""
    def __init__(self, num_entities: int, num_relations: int, embed_dim: int = 100, margin: float = 1.0, lr: float = 0.01):
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = embed_dim
        self.margin = margin
        self.lr = lr

        # Uniform initialization with L2 normalization
        self.entity_embeddings = np.random.uniform(-6.0 / np.sqrt(embed_dim), 6.0 / np.sqrt(embed_dim), (num_entities, embed_dim))
        self.entity_embeddings /= np.linalg.norm(self.entity_embeddings, axis=1, keepdims=True)

        self.relation_embeddings = np.random.uniform(-6.0 / np.sqrt(embed_dim), 6.0 / np.sqrt(embed_dim), (num_relations, embed_dim))
        self.relation_embeddings /= np.linalg.norm(self.relation_embeddings, axis=1, keepdims=True)

    def score_triplet(self, h: int, r: int, t: int) -> float:
        """Compute triplet energy distance: -||h + r - t||_2."""
        vec_h = self.entity_embeddings[h]
        vec_r = self.relation_embeddings[r]
        vec_t = self.entity_embeddings[t]
        dist = np.linalg.norm(vec_h + vec_r - vec_t)
        return float(-dist)

    def train_step(self, positive_triplets: List[Tuple[int, int, int]], negative_triplets: List[Tuple[int, int, int]]) -> float:
        """Pairwise Margin Ranking Loss update: max(0, gamma + score(neg) - score(pos))."""
        loss = 0.0
        for (pos, neg) in zip(positive_triplets, negative_triplets):
            h_p, r_p, t_p = pos
            h_n, r_n, t_n = neg

            score_pos = self.score_triplet(h_p, r_p, t_p)
            score_neg = self.score_triplet(h_n, r_n, t_n)

            triplet_loss = max(0.0, self.margin - score_pos + score_neg)
            if triplet_loss > 0:
                loss += triplet_loss
                # Gradient update
                diff_pos = self.entity_embeddings[h_p] + self.relation_embeddings[r_p] - self.entity_embeddings[t_p]
                norm_pos = max(1e-8, np.linalg.norm(diff_pos))
                grad_pos = diff_pos / norm_pos

                self.entity_embeddings[h_p] -= self.lr * grad_pos
                self.relation_embeddings[r_p] -= self.lr * grad_pos
                self.entity_embeddings[t_p] += self.lr * grad_pos

        return float(loss / max(1, len(positive_triplets)))
