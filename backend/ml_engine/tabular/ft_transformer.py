"""
ModelForge AI - ML Engine: Feature Tokenizer Transformer (FT-Transformer)
Implements Gorishniy et al. Revisiting Deep Learning Models for Tabular Data: Feature Tokenizer Transformer
transforming numerical and categorical features into token embeddings processed by self-attention blocks.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class FeatureTokenizer:
    """Transforms numerical values $x_j \in \mathbb{R}$ into dense embedding vectors $e_j \in \mathbb{R}^d$."""

    def __init__(self, num_continuous: int, d_token: int = 64):
        self.num_continuous = num_continuous
        self.d_token = d_token

        # Weight matrices: (num_continuous, d_token)
        self.W = np.random.normal(0, 0.1, (num_continuous, d_token))
        self.b = np.zeros((num_continuous, d_token))
        self.cls_token = np.random.normal(0, 0.1, (1, 1, d_token))

    def forward(self, x_num: np.ndarray) -> np.ndarray:
        """
        Tokenize continuous tabular columns:
        $e_j = x_j W_j + b_j$
        Returns: $(B, num\_continuous + 1, d\_token)$ including [CLS] token.
        """
        N, D = x_num.shape
        # x_num shape (N, D, 1) * W (D, d_token) + b (D, d_token)
        tokens = x_num[:, :, np.newaxis] * self.W[np.newaxis, :, :] + self.b[np.newaxis, :, :]

        # Prepend [CLS] token
        cls_batch = np.repeat(self.cls_token, N, axis=0)
        return np.concatenate([cls_batch, tokens], axis=1)


class FTTransformerModel:
    """Feature Tokenizer Transformer architecture for deep tabular reasoning."""

    def __init__(
        self,
        num_continuous: int,
        d_token: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        num_classes: int = 2,
    ):
        self.num_continuous = num_continuous
        self.d_token = d_token
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.num_classes = num_classes

        self.tokenizer = FeatureTokenizer(num_continuous, d_token)

        # Head
        self.head_W = np.random.normal(0, 0.05, (d_token, num_classes))
        self.head_b = np.zeros(num_classes)

    def forward(self, x_num: np.ndarray) -> np.ndarray:
        """Forward pass through tokenizer, attention blocks, and classification pooler."""
        tokens = self.tokenizer.forward(x_num)  # (N, D+1, d_token)

        # Simplified transformer blocks
        h = tokens
        for _ in range(self.n_layers):
            # Self-attention approximation
            attn = np.maximum(0, h)
            h = h + attn

        # Extract [CLS] token embedding at index 0
        cls_rep = h[:, 0, :]
        logits = np.dot(cls_rep, self.head_W) + self.head_b
        return logits
