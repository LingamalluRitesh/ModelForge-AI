"""
ModelForge AI - Feature Store: In-Memory Fast Vector Index & Entity Lookup
Stores normalized dense entity vectors and executes sub-millisecond Nearest Neighbor searches.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class InMemoryVectorIndex:
    """In-memory cosine similarity vector index."""
    def __init__(self, dimension: int = 512):
        self.dim = dimension
        self.entity_keys: List[str] = []
        self.vectors: List[np.ndarray] = []

    def insert(self, entity_key: str, vector: np.ndarray):
        vec = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        self.entity_keys.append(entity_key)
        self.vectors.append(vec)

    def query(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        if not self.vectors:
            return []
        q_vec = np.asarray(query_vector, dtype=np.float32)
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm

        matrix = np.vstack(self.vectors)
        scores = np.dot(matrix, q_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(self.entity_keys[i], float(scores[i])) for i in top_indices]
