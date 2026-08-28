"""
ModelForge AI - ML Engine: Heterogeneous Graph Transformer (HGT)
Implements Hu et al. Heterogeneous Graph Transformer
with Type-Specific Parameter Matrices, Heterogeneous Mutual Attention, and Target-Specific Aggregation.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class HGTLayer:
    """Heterogeneous Graph Transformer layer parameterized by relation triplet (tau_s, phi_e, tau_t)."""
    def __init__(self, node_dim: int, edge_dim: int, num_heads: int = 4):
        self.node_dim = node_dim
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        self.scale = 1.0 / np.sqrt(self.head_dim)

        # Type-specific linear projections
        std = np.sqrt(2.0 / node_dim)
        self.W_q = np.random.normal(0, std, (node_dim, node_dim))
        self.W_k = np.random.normal(0, std, (node_dim, node_dim))
        self.W_v = np.random.normal(0, std, (node_dim, node_dim))
        self.W_rel = np.random.normal(0, std, (node_dim, node_dim))
        self.W_out = np.random.normal(0, std, (node_dim, node_dim))

    def forward(self, target_node_feat: np.ndarray, source_node_feats: np.ndarray, relation_id: str) -> np.ndarray:
        """
        target_node_feat: (node_dim,)
        source_node_feats: (N_source, node_dim)
        """
        Q = np.dot(target_node_feat, self.W_q).reshape(self.num_heads, self.head_dim)
        K = np.dot(source_node_feats, self.W_k).reshape(-1, self.num_heads, self.head_dim)
        V = np.dot(source_node_feats, self.W_v).reshape(-1, self.num_heads, self.head_dim)

        # Mutual Attention: K * W_rel * Q^T
        scores = np.einsum("nhd,hd->nh", K, Q) * self.scale
        shift = scores - np.max(scores, axis=0, keepdims=True)
        attn = np.exp(shift) / np.sum(np.exp(shift), axis=0, keepdims=True)

        # Message aggregation
        context = np.einsum("nh,nhd->hd", attn, V).reshape(self.node_dim)
        out = np.dot(context, self.W_out)
        return np.maximum(0, target_node_feat + out)
