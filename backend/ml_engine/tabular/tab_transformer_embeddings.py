"""
ModelForge AI - ML Engine: TabTransformer Contextual Embeddings
Implements Huang et al. TabTransformer: Tabular Data Modeling Using Contextual Embeddings
transforming categorical column embeddings into robust contextual representations via multi-head self-attention.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class TabTransformerColumnAttention:
    """Multi-head self-attention layer across categorical column token representations."""
    def __init__(self, num_categories: int, embed_dim: int = 32, num_heads: int = 4):
        self.num_cats = num_categories
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / np.sqrt(self.head_dim)

        std = np.sqrt(2.0 / embed_dim)
        self.W_q = np.random.normal(0, std, (embed_dim, embed_dim))
        self.W_k = np.random.normal(0, std, (embed_dim, embed_dim))
        self.W_v = np.random.normal(0, std, (embed_dim, embed_dim))
        self.W_out = np.random.normal(0, std, (embed_dim, embed_dim))

    def forward(self, cat_embeddings: np.ndarray) -> np.ndarray:
        """
        cat_embeddings: (B, num_cats, embed_dim)
        """
        B, M, D = cat_embeddings.shape
        Q = np.dot(cat_embeddings, self.W_q).reshape(B, M, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = np.dot(cat_embeddings, self.W_k).reshape(B, M, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = np.dot(cat_embeddings, self.W_v).reshape(B, M, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) * self.scale
        shift = scores - np.max(scores, axis=-1, keepdims=True)
        attn = np.exp(shift) / np.sum(np.exp(shift), axis=-1, keepdims=True)

        context = np.matmul(attn, V).transpose(0, 2, 1, 3).reshape(B, M, D)
        out = np.dot(context, self.W_out)
        return cat_embeddings + out


class TabTransformerModel:
    """Complete TabTransformer combining contextual categorical embeddings with continuous features."""
    def __init__(
        self,
        num_categorical: int,
        num_continuous: int,
        embed_dim: int = 32,
        num_layers: int = 3,
        mlp_hidden_dims: List[int] = [128, 64],
        num_classes: int = 1,
    ):
        self.num_cats = num_categorical
        self.num_conts = num_continuous
        self.embed_dim = embed_dim

        self.cat_embeddings = np.random.normal(0, 0.1, (num_categorical, embed_dim))
        self.transformer_blocks = [TabTransformerColumnAttention(num_categorical, embed_dim) for _ in range(num_layers)]

        # Continuous layer norm parameters
        self.cont_gamma = np.ones(num_continuous)
        self.cont_beta = np.zeros(num_continuous)

        in_dim = num_categorical * embed_dim + num_continuous
        self.mlp_weights = []
        self.mlp_biases = []
        curr_dim = in_dim
        for h in mlp_hidden_dims:
            self.mlp_weights.append(np.random.normal(0, np.sqrt(2.0 / curr_dim), (curr_dim, h)))
            self.mlp_biases.append(np.zeros(h))
            curr_dim = h

        self.head_W = np.random.normal(0, np.sqrt(2.0 / curr_dim), (curr_dim, num_classes))
        self.head_b = np.zeros(num_classes)

    def forward(self, x_cat: np.ndarray, x_cont: np.ndarray) -> np.ndarray:
        B = x_cat.shape[0]
        # Embed categoricals
        cat_tokens = x_cat[:, :, np.newaxis] * self.cat_embeddings[np.newaxis, :, :]  # (B, M, D)

        # Contextual Self-Attention blocks
        h_cat = cat_tokens
        for block in self.transformer_blocks:
            h_cat = block.forward(h_cat)
        flat_cat = h_cat.reshape(B, -1)

        # Continuous normalization
        mean = np.mean(x_cont, axis=0, keepdims=True)
        std = np.std(x_cont, axis=0, keepdims=True) + 1e-5
        norm_cont = ((x_cont - mean) / std) * self.cont_gamma + self.cont_beta

        # Concatenate and pass through MLP
        combined = np.concatenate([flat_cat, norm_cont], axis=-1)
        h = combined
        for W, b in zip(self.mlp_weights, self.mlp_biases):
            h = np.maximum(0, np.dot(h, W) + b)

        logits = np.dot(h, self.head_W) + self.head_b
        return logits
