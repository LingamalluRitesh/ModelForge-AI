"""
ModelForge AI - ML Engine: TabTransformer (Tabular Data Modeling Using Contextual Embeddings)
Implements Huang et al. TabTransformer: Tabular Data Modeling Using Contextual Embeddings
mapping categorical column features through multi-head self-attention to learn robust contextual embeddings
invariant to missing values and noisy feature distributions.
$\mathbf{E}_\phi(x_{cat}) = \text{Transformer}(\mathbf{E}(x_{cat})) = [\mathbf{e}_{\phi, 1}, \mathbf{e}_{\phi, 2}, \dots, \mathbf{e}_{\phi, m}]$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class TabTransformerAttentionBlock:
    """Multi-head Self-Attention block across categorical column tokens."""

    def __init__(self, embed_dim: int = 32, num_heads: int = 4, ffn_dim: int = 64, dropout_rate: float = 0.1):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / np.sqrt(self.head_dim)
        self.dropout_rate = dropout_rate

        std = np.sqrt(2.0 / embed_dim)
        self.W_q = np.random.normal(0, std, (embed_dim, embed_dim))
        self.W_k = np.random.normal(0, std, (embed_dim, embed_dim))
        self.W_v = np.random.normal(0, std, (embed_dim, embed_dim))
        self.W_out = np.random.normal(0, std, (embed_dim, embed_dim))

        # Layer Normalization 1
        self.ln1_gamma = np.ones(embed_dim)
        self.ln1_beta = np.zeros(embed_dim)

        # Feed-Forward Network
        self.ffn_w1 = np.random.normal(0, np.sqrt(2.0 / embed_dim), (embed_dim, ffn_dim))
        self.ffn_b1 = np.zeros(ffn_dim)
        self.ffn_w2 = np.random.normal(0, np.sqrt(2.0 / ffn_dim), (ffn_dim, embed_dim))
        self.ffn_b2 = np.zeros(embed_dim)

        # Layer Normalization 2
        self.ln2_gamma = np.ones(embed_dim)
        self.ln2_beta = np.zeros(embed_dim)

    def _layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True) + 1e-5
        return gamma * ((x - mean) / np.sqrt(var)) + beta

    def forward(self, x: np.ndarray, is_training: bool = True) -> np.ndarray:
        """
        x: (B, num_cats, embed_dim)
        """
        B, M, D = x.shape
        # 1. Multi-Head Self-Attention
        norm1 = self._layer_norm(x, self.ln1_gamma, self.ln1_beta)
        Q = np.dot(norm1, self.W_q).reshape(B, M, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = np.dot(norm1, self.W_k).reshape(B, M, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = np.dot(norm1, self.W_v).reshape(B, M, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) * self.scale
        shift = scores - np.max(scores, axis=-1, keepdims=True)
        attn = np.exp(shift) / np.sum(np.exp(shift), axis=-1, keepdims=True)

        context = np.matmul(attn, V).transpose(0, 2, 1, 3).reshape(B, M, D)
        attn_out = np.dot(context, self.W_out)
        h1 = x + attn_out

        # 2. Feed-Forward Network
        norm2 = self._layer_norm(h1, self.ln2_gamma, self.ln2_beta)
        ffn_h = np.maximum(0, np.dot(norm2, self.ffn_w1) + self.ffn_b1)
        ffn_out = np.dot(ffn_h, self.ffn_w2) + self.ffn_b2
        out = h1 + ffn_out
        return out


class TabTransformer:
    """Full TabTransformer combining self-attended contextual categorical columns with continuous features."""

    def __init__(
        self,
        num_categories_list: List[int],
        num_continuous: int,
        embed_dim: int = 32,
        num_layers: int = 3,
        num_heads: int = 4,
        mlp_hidden_dims: List[int] = [128, 64],
        num_classes: int = 1,
    ):
        self.num_cats = len(num_categories_list)
        self.num_conts = num_continuous
        self.embed_dim = embed_dim

        # Per-column categorical embedding dictionaries
        self.cat_embeddings = [
            np.random.normal(0, 0.1, (cardinality + 1, embed_dim)) for cardinality in num_categories_list
        ]

        # Stack of Transformer blocks
        self.blocks = [
            TabTransformerAttentionBlock(embed_dim, num_heads, embed_dim * 2) for _ in range(num_layers)
        ]

        # Continuous feature normalization layer
        self.cont_gamma = np.ones(num_continuous) if num_continuous > 0 else np.array([])
        self.cont_beta = np.zeros(num_continuous) if num_continuous > 0 else np.array([])

        # Top MLP Predictor
        in_dim = (self.num_cats * embed_dim) + num_continuous
        self.mlp_weights = []
        self.mlp_biases = []
        curr = in_dim
        for h in mlp_hidden_dims:
            self.mlp_weights.append(np.random.normal(0, np.sqrt(2.0 / curr), (curr, h)))
            self.mlp_biases.append(np.zeros(h))
            curr = h

        self.head_W = np.random.normal(0, np.sqrt(2.0 / curr), (curr, num_classes))
        self.head_b = np.zeros(num_classes)

    def forward(self, x_cat: np.ndarray, x_cont: Optional[np.ndarray] = None, is_training: bool = True) -> np.ndarray:
        B = x_cat.shape[0]
        # Embed categorical columns
        cat_tokens = []
        for j in range(self.num_cats):
            col_indices = np.clip(x_cat[:, j].astype(int), 0, len(self.cat_embeddings[j]) - 1)
            cat_tokens.append(self.cat_embeddings[j][col_indices])  # (B, D)

        h_cat = np.stack(cat_tokens, axis=1)  # (B, M, D)

        # Pass through Transformer blocks
        for block in self.blocks:
            h_cat = block.forward(h_cat, is_training=is_training)

        flat_cat = h_cat.reshape(B, -1)

        # Normalize continuous features
        if x_cont is not None and self.num_conts > 0:
            mean = np.mean(x_cont, axis=0, keepdims=True)
            std = np.std(x_cont, axis=0, keepdims=True) + 1e-5
            norm_cont = self.cont_gamma * ((x_cont - mean) / std) + self.cont_beta
            features = np.concatenate([flat_cat, norm_cont], axis=-1)
        else:
            features = flat_cat

        # Pass through MLP
        h = features
        for W, b in zip(self.mlp_weights, self.mlp_biases):
            h = np.maximum(0, np.dot(h, W) + b)

        logits = np.dot(h, self.head_W) + self.head_b
        return logits
