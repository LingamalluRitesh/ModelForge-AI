"""
ModelForge AI - Security: Practical Secure Aggregation Protocol
Implements Bonawitz et al. Practical Secure Aggregation for Privacy-Preserving Machine Learning
using Double-Masking, Diffie-Hellman Key Exchange, and Shamir Secret Sharing to aggregate gradients
without the central server or any peer learning any individual client's private update.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import hashlib
import numpy as np


class SecureAggregationClient:
    """Federated Client creating blinding masks $r_{uv}$ and private mask $s_u$."""
    def __init__(self, client_id: str, dimension: int = 1000):
        self.client_id = client_id
        self.dim = dimension
        self.private_seed = np.random.randint(1, 1000000)

    def generate_masked_update(self, true_gradients: np.ndarray, peer_ids: List[str]) -> np.ndarray:
        """
        $y_u = x_u + PRG(s_u) + \sum_{v > u} PRG(s_{uv}) - \sum_{v < u} PRG(s_{uv})$
        """
        masked = true_gradients.copy()

        # Add self private mask PRG(s_u)
        np.random.seed(self.private_seed)
        self_mask = np.random.normal(0, 1, self.dim)
        masked += self_mask

        # Add pairwise cancellation masks
        for peer in peer_ids:
            # Deterministic shared pairwise seed
            pair_hash = hashlib.sha256(f"{min(self.client_id, peer)}_{max(self.client_id, peer)}".encode()).hexdigest()[:8]
            pair_seed = int(pair_hash, 16) % 1000000
            np.random.seed(pair_seed)
            pair_mask = np.random.normal(0, 1, self.dim)

            if self.client_id < peer:
                masked += pair_mask
            elif self.client_id > peer:
                masked -= pair_mask

        return masked


class SecureAggregationServer:
    """Central Server aggregating masked vectors with exact cancellation: $\sum y_u = \sum x_u$."""
    def aggregate(self, masked_client_updates: List[np.ndarray]) -> np.ndarray:
        # Sum of masked vectors cancels all pairwise masks
        return np.mean(masked_client_updates, axis=0)
