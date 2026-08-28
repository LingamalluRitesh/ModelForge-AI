"""
ModelForge AI - ML Engine: FlashAttention Exact Memory-Efficient Algorithm
Implements Dao et al. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
using SRAM Tiling, Online Softmax Normalization, and Block-Sparse Tiled Matrix Multiplications.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class FlashAttentionTiling:
    """Computes exact scaled dot-product attention in $O(N)$ SRAM memory via block tiling."""
    def __init__(self, block_size_q: int = 32, block_size_kv: int = 32):
        self.B_r = block_size_q
        self.B_c = block_size_kv

    def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, scale: float) -> np.ndarray:
        """
        Q, K, V shapes: (B, num_heads, N, d)
        """
        B, H, N, d = Q.shape
        O = np.zeros_like(Q)
        l = np.zeros((B, H, N, 1))           # Row sum normalizer
        m = np.full((B, H, N, 1), -np.inf)   # Row max values for online softmax stability

        Tr = (N + self.B_r - 1) // self.B_r
        Tc = (N + self.B_c - 1) // self.B_c

        # Outer loop over key-value blocks (Column tiles)
        for j in range(Tc):
            kv_start = j * self.B_c
            kv_end = min(N, (j + 1) * self.B_c)
            K_j = K[:, :, kv_start:kv_end, :]  # (B, H, B_c, d)
            V_j = V[:, :, kv_start:kv_end, :]  # (B, H, B_c, d)

            # Inner loop over query blocks (Row tiles)
            for i in range(Tr):
                q_start = i * self.B_r
                q_end = min(N, (i + 1) * self.B_r)
                Q_i = Q[:, :, q_start:q_end, :]  # (B, H, B_r, d)

                # Compute local block dot product: S_ij = Q_i K_j^T * scale
                S_ij = np.matmul(Q_i, K_j.transpose(0, 1, 3, 2)) * scale

                # Online softmax statistics update
                m_ij = np.max(S_ij, axis=-1, keepdims=True)
                P_ij = np.exp(S_ij - m_ij)
                l_ij = np.sum(P_ij, axis=-1, keepdims=True)

                m_prev = m[:, :, q_start:q_end, :]
                l_prev = l[:, :, q_start:q_end, :]

                m_new = np.maximum(m_prev, m_ij)
                l_new = np.exp(m_prev - m_new) * l_prev + np.exp(m_ij - m_new) * l_ij

                # Update output accumulator
                O_prev = O[:, :, q_start:q_end, :]
                O_new = (
                    np.exp(m_prev - m_new) * l_prev * O_prev
                    + np.exp(m_ij - m_new) * np.matmul(P_ij, V_j)
                ) / np.maximum(1e-10, l_new)

                O[:, :, q_start:q_end, :] = O_new
                m[:, :, q_start:q_end, :] = m_new
                l[:, :, q_start:q_end, :] = l_new

        return O
