"""
ModelForge AI - ML Engine: Heterogeneous Graph Transformer (HGT) Architecture
Implements Hu et al. Heterogeneous Graph Transformer
with type-specific Query/Key/Value projections, relation-aware mutual attention, and relative temporal Bochner kernels.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class HeterogeneousHGTLayer:
    """Type-parameterized multi-head mutual attention for heterogeneous knowledge networks."""

    def __init__(
        self,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],  # (src_type, rel_type, tgt_type)
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
    ):
        self.node_types = node_types
        self.edge_types = edge_types
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.d_k = out_dim // num_heads
        self.scale = 1.0 / np.sqrt(self.d_k)

        # Per-node-type linear projections
        std = np.sqrt(2.0 / in_dim)
        self.W_q = {nt: np.random.normal(0, std, (in_dim, out_dim)) for nt in node_types}
        self.W_k = {nt: np.random.normal(0, std, (in_dim, out_dim)) for nt in node_types}
        self.W_v = {nt: np.random.normal(0, std, (in_dim, out_dim)) for nt in node_types}

        # Relation-specific transformation matrices W_rel (num_heads, d_k, d_k)
        self.W_rel = {
            et: np.random.normal(0, np.sqrt(2.0 / self.d_k), (num_heads, self.d_k, self.d_k)) for et in edge_types
        }

    def compute_mutual_attention(
        self,
        src_features: np.ndarray,
        tgt_features: np.ndarray,
        edge_type: Tuple[str, str, str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        src_features: (N_src, in_dim)
        tgt_features: (N_tgt, in_dim)
        """
        src_type, rel, tgt_type = edge_type
        N_src = len(src_features)
        N_tgt = len(tgt_features)

        # Project Q and K
        Q = np.dot(tgt_features, self.W_q[tgt_type]).reshape(N_tgt, self.num_heads, self.d_k)
        K = np.dot(src_features, self.W_k[src_type]).reshape(N_src, self.num_heads, self.d_k)
        V = np.dot(src_features, self.W_v[src_type]).reshape(N_src, self.num_heads, self.d_k)

        # Relation transformed Key: K * W_rel
        W_r = self.W_rel[edge_type]  # (H, d_k, d_k)
        K_rel = np.einsum("nhd,hde->nhe", K, W_r)

        # Multi-Head Attention scores: Q * K_rel^T
        scores = np.einsum("thd,shd->hts", Q, K_rel) * self.scale
        shift = scores - np.max(scores, axis=-1, keepdims=True)
        attn = np.exp(shift) / np.sum(np.exp(shift), axis=-1, keepdims=True)

        # Message passing: Attn * V
        messages = np.einsum("hts,shd->thd", attn, V).reshape(N_tgt, self.out_dim)
        return messages, attn
