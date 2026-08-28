"""
ModelForge AI - ML Engine: Scaled Dot-Product Multi-Head Self-Attention
Implements Vaswani et al. Multi-Head Attention, Rotary Positional Embeddings (RoPE),
Causal Autoregressive Masking, and Key-Value (KV) Caching for sequence decoding.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class RotaryPositionalEmbedding:
    """Su et al. Rotary Position Embedding (RoPE) for relative position awareness."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute theta frequencies: theta_i = 10000^(-2(i-1)/dim)
        inv_freq = 1.0 / (base ** (np.arange(0, dim, 2).astype(np.float32) / dim))
        t = np.arange(max_seq_len, dtype=np.float32)
        freqs = np.outer(t, inv_freq)  # shape (max_seq_len, dim // 2)

        self.cos_cached = np.cos(freqs)  # shape (max_seq_len, dim // 2)
        self.sin_cached = np.sin(freqs)

    def apply_rope(self, x: np.ndarray, seq_len: int) -> np.ndarray:
        """Apply 2D complex plane rotation to pairs of vector elements."""
        # x shape: (batch_size, seq_len, num_heads, head_dim)
        B, S, H, D = x.shape
        half_d = D // 2

        x1 = x[:, :, :, :half_d]
        x2 = x[:, :, :, half_d:]

        cos = self.cos_cached[:S, None, :]  # shape (S, 1, half_d)
        sin = self.sin_cached[:S, None, :]

        # Rotation: [x1 * cos - x2 * sin, x1 * sin + x2 * cos]
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        return np.concatenate([out1, out2], axis=-1)


class MultiHeadSelfAttention:
    """Multi-Head Scaled Dot-Product Attention with causal masking and KV-Cache."""

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        use_rope: bool = True,
        dropout_rate: float = 0.1,
    ):
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / np.sqrt(self.head_dim)
        self.use_rope = use_rope
        self.dropout_rate = dropout_rate

        # Weight matrices
        std = np.sqrt(2.0 / embed_dim)
        self.W_q = np.random.normal(0, std, (embed_dim, embed_dim))
        self.W_k = np.random.normal(0, std, (embed_dim, embed_dim))
        self.W_v = np.random.normal(0, std, (embed_dim, embed_dim))
        self.W_o = np.random.normal(0, std, (embed_dim, embed_dim))

        self.rope = RotaryPositionalEmbedding(self.head_dim) if use_rope else None

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        shift_x = x - np.max(x, axis=-1, keepdims=True)
        exps = np.exp(shift_x)
        return exps / np.sum(exps, axis=-1, keepdims=True)

    def forward(
        self,
        x: np.ndarray,
        is_causal: bool = False,
        kv_cache: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]], np.ndarray]:
        """
        Forward multi-head attention:
        $Attention(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$
        """
        B, S, E = x.shape

        # Linear projections
        Q = np.dot(x, self.W_q).reshape(B, S, self.num_heads, self.head_dim)
        K = np.dot(x, self.W_k).reshape(B, S, self.num_heads, self.head_dim)
        V = np.dot(x, self.W_v).reshape(B, S, self.num_heads, self.head_dim)

        if self.use_rope and self.rope:
            Q = self.rope.apply_rope(Q, S)
            K = self.rope.apply_rope(K, S)

        # Handle KV Caching
        if kv_cache is not None:
            if "k" in kv_cache and "v" in kv_cache:
                K = np.concatenate([kv_cache["k"], K], axis=1)
                V = np.concatenate([kv_cache["v"], V], axis=1)
            kv_cache = {"k": K, "v": V}

        # Transpose for batched matrix multiplication: (B, num_heads, S, head_dim)
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)

        # Scaled dot product scores: (B, num_heads, S_q, S_k)
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) * self.scale

        # Causal lower-triangular mask
        if is_causal:
            S_q = Q.shape[2]
            S_k = K.shape[2]
            mask = np.triu(np.full((S_q, S_k), -1e9), k=1)
            scores = scores + mask

        # Attention weights
        attn_weights = self._softmax(scores)

        # Context output
        context = np.matmul(attn_weights, V)  # (B, num_heads, S_q, head_dim)
        context = context.transpose(0, 2, 1, 3).reshape(B, S, E)

        # Final linear out projection
        out = np.dot(context, self.W_o)

        return out, kv_cache, attn_weights
