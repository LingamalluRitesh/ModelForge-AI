"""
ModelForge AI - ML Engine: Relational Graph Convolutional Networks (RGCN)
Implements Schlichtkrull et al. Modeling Relational Data with Graph Convolutional Networks
for multi-relational Knowledge Graphs, syndicate fraud rings, and heterogeneous entity graphs.
$h_i^{(l+1)} = \sigma\left(W_0^{(l)} h_i^{(l)} + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_i^r} \frac{1}{c_{i,r}} W_r^{(l)} h_j^{(l)}\right)$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class RelationalGraphConvLayer:
    """Multi-relational Graph Convolutional layer with basis decomposition."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_relations: int,
        num_bases: Optional[int] = None,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        self.num_bases = num_bases or min(num_relations, 8)

        # Basis matrices: (num_bases, in_features, out_features)
        std_b = np.sqrt(2.0 / in_features)
        self.bases = np.random.normal(0, std_b, (self.num_bases, in_features, out_features))

        # Relation coefficients: (num_relations, num_bases)
        self.rel_coeffs = np.random.normal(0, 0.1, (num_relations, self.num_bases))

        # Self-loop root transformation
        self.W_root = np.random.normal(0, std_b, (in_features, out_features))
        self.bias = np.zeros(out_features)

    def _get_relation_weight(self, rel_id: int) -> np.ndarray:
        """Construct $W_r = \sum_b a_{rb} V_b$ via basis decomposition."""
        coeffs = self.rel_coeffs[rel_id]  # (num_bases,)
        # Linear combination of bases
        W_r = np.tensordot(coeffs, self.bases, axes=(0, 0))
        return W_r

    def forward(
        self,
        node_features: np.ndarray,
        adjacency_by_relation: Dict[int, np.ndarray],
    ) -> np.ndarray:
        """
        Forward RGCN pass:
        node_features: $(N, in\_features)$
        adjacency_by_relation: mapping of rel_id -> normalized adjacency matrix $(N, N)$
        """
        N = node_features.shape[0]

        # 1. Root self-connection: W_0 * h_i
        out = np.dot(node_features, self.W_root)

        # 2. Accumulate message passing per relation type: sum_r A_r * H * W_r
        for rel_id, adj in adjacency_by_relation.items():
            if rel_id >= self.num_relations:
                continue

            W_r = self._get_relation_weight(rel_id)
            rel_msg = np.dot(node_features, W_r)
            # Neighborhood aggregation
            rel_agg = np.dot(adj, rel_msg)
            out += rel_agg

        # Add bias & ReLU
        out = np.maximum(0, out + self.bias)
        return out
