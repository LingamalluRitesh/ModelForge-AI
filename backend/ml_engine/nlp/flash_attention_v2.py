"""
ModelForge AI - NLP Engine: FlashAttention-2 (Fast and Memory-Efficient Exact Attention with IO-Awareness)
Implements Dao FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning
optimizing block matrix multiplication scheduling across fast GPU SRAM tiles and eliminating quadratic memory footprint.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class FlashAttention2Engine:
    """Tiled block-sparse exact self-attention computing online softmax scaling."""

    def __init__(self, block_size_row: int = 64, block_size_col: int = 64):
        self.Br = block_size_row
        self.Bc = block_size_col

    def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, is_causal: bool = False) -> np.ndarray:
        """
        Q: (B, num_heads, N, d)
        K: (B, num_heads, N, d)
        V: (B, num_heads, N, d)
        """
        B, H, N, d = Q.shape
        scale = 1.0 / np.sqrt(d)
        O = np.zeros_like(Q)

        for b in range(B):
            for h in range(H):
                q = Q[b, h]
                k = K[b, h]
                v = V[b, h]

                # Initialize running statistics per row tile
                m = np.full((N,), -np.inf)
                l = np.zeros((N,))
                out = np.zeros((N, d))

                # Loop over column blocks (Keys / Values)
                for j in range(0, N, self.Bc):
                    k_block = k[j : j + self.Bc]
                    v_block = v[j : j + self.Bc]

                    # Loop over row blocks (Queries)
                    for i in range(0, N, self.Br):
                        q_block = q[i : i + self.Br]

                        # Block similarity: S_ij = Q_i * K_j^T * scale
                        s_ij = np.dot(q_block, k_block.T) * scale

                        if is_causal:
                            # Apply causal masking
                            row_idx = np.arange(i, min(i + self.Br, N))[:, None]
                            col_idx = np.arange(j, min(j + self.Bc, N))[None, :]
                            s_ij = np.where(row_idx >= col_idx, s_ij, -np.inf)

                        # Row-wise max of current block
                        m_ij = np.max(s_ij, axis=-1)
                        p_ij = np.exp(s_ij - m_ij[:, None])
                        l_ij = np.sum(p_ij, axis=-1)

                        # Update online softmax statistics
                        m_prev = m[i : i + self.Br]
                        l_prev = l[i : i + self.Br]
                        m_new = np.maximum(m_prev, m_ij)

                        alpha = np.exp(m_prev - m_new)
                        beta = np.exp(m_ij - m_new)

                        l_new = alpha * l_prev + beta * l_ij

                        # Update accumulator
                        pv = np.dot(p_ij, v_block)
                        out[i : i + self.Br] = (
                            (alpha[:, None] * l_prev[:, None] * out[i : i + self.Br])
                            + (beta[:, None] * pv)
                        ) / np.maximum(1e-8, l_new[:, None])

                        m[i : i + self.Br] = m_new
                        l[i : i + self.Br] = l_new

                O[b, h] = out

        return O
