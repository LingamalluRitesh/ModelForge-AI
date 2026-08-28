"""
ModelForge AI - ML Engine: Graph Attention Networks (GAT)
Implements Velickovic et al. Graph Attention Networks with Multi-Head Self-Attention,
LeakyReLU anisotropic edge coefficients, and neighborhood aggregation for graph representation.
$\alpha_{ij} = \frac{\exp(\text{LeakyReLU}(\mathbf{a}^T [\mathbf{W} h_i || \mathbf{W} h_j]))}{\sum_{k \in \mathcal{N}_i} \exp(\text{LeakyReLU}(\mathbf{a}^T [\mathbf{W} h_i || \mathbf{W} h_k]))}$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class GraphAttentionLayer:
    """Individual GAT layer with multi-head attention over graph adjacency."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int = 4,
        leaky_relu_slope: float = 0.2,
        concat_heads: bool = True,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.leaky_slope = leaky_relu_slope
        self.concat_heads = concat_heads

        # Weight matrices: (n_heads, in_features, out_features)
        std = np.sqrt(2.0 / in_features)
        self.W = np.random.normal(0, std, (n_heads, in_features, out_features))

        # Attention parameter vectors: a_src and a_dst of shape (n_heads, out_features)
        std_a = np.sqrt(2.0 / out_features)
        self.a_src = np.random.normal(0, std_a, (n_heads, out_features))
        self.a_dst = np.random.normal(0, std_a, (n_heads, out_features))

    def _leaky_relu(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.leaky_slope * x)

    def _softmax(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Softmax only over non-zero adjacency connections."""
        x_masked = np.where(mask, x, -1e9)
        shift_x = x_masked - np.max(x_masked, axis=-1, keepdims=True)
        exps = np.where(mask, np.exp(shift_x), 0.0)
        sum_exps = np.sum(exps, axis=-1, keepdims=True)
        return exps / np.maximum(1e-10, sum_exps)

    def forward(self, H: np.ndarray, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward GAT layer:
        H: Node features matrix $(N, in\_features)$
        A: Graph adjacency matrix $(N, N)$ with self-loops
        """
        N = H.shape[0]

        # 1. Add self-loops to adjacency
        A_loop = A.copy()
        np.fill_diagonal(A_loop, 1.0)
        adj_mask = A_loop > 0  # (N, N)

        # 2. Linear projection for all heads: Wh shape (n_heads, N, out_features)
        Wh = np.matmul(H[np.newaxis, :, :], self.W)

        # 3. Compute attention logits: e_ij = a_src^T Wh_i + a_dst^T Wh_j
        # Shape: (n_heads, N, 1) and (n_heads, 1, N)
        attn_src = np.sum(Wh * self.a_src[:, np.newaxis, :], axis=-1, keepdims=True)
        attn_dst = np.sum(Wh * self.a_dst[:, np.newaxis, :], axis=-1)[:, np.newaxis, :]

        e = attn_src + attn_dst  # Shape (n_heads, N, N)
        e = self._leaky_relu(e)

        # 4. Normalized attention coefficients
        alpha = np.zeros((self.n_heads, N, N))
        for h in range(self.n_heads):
            alpha[h] = self._softmax(e[h], adj_mask)

        # 5. Neighborhood aggregation: H_new = alpha * Wh
        # Shape (n_heads, N, out_features)
        out_heads = np.matmul(alpha, Wh)

        if self.concat_heads:
            # Concatenate along feature axis: (N, n_heads * out_features)
            H_out = out_heads.transpose(1, 0, 2).reshape(N, -1)
        else:
            # Average across heads: (N, out_features)
            H_out = np.mean(out_heads, axis=0)

        return H_out, alpha
