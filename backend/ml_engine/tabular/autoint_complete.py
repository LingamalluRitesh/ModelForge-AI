"""
ModelForge AI - ML Engine: AutoInt Complete Multi-Head Attention Architecture
Implements Song et al. AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks
modeling explicit high-order non-linear feature interactions in tabular datasets.
$\mathbf{z}_i^{(l+1)} = \sum_{j=1}^M lpha_{i,j}^{(l)} (\mathbf{W}_V^{(l)} \mathbf{z}_j^{(l)}) + \mathbf{W}_{Res}^{(l)} \mathbf{z}_i^{(l)}$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class MultiHeadTabularSelfAttention:
    """Multi-Head Self-Attention for tabular feature interaction mapping."""
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout_rate: float = 0.1):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / np.sqrt(self.head_dim)

        std = np.sqrt(2.0 / embed_dim)
        self.W_q = np.random.normal(0, std, (embed_dim, embed_dim))
        self.W_k = np.random.normal(0, std, (embed_dim, embed_dim))
        self.W_v = np.random.normal(0, std, (embed_dim, embed_dim))
        self.W_res = np.random.normal(0, std, (embed_dim, embed_dim))
        self.dropout_rate = dropout_rate

    def forward(self, x: np.ndarray) -> np.ndarray:
        B, M, D = x.shape
        Q = np.dot(x, self.W_q).reshape(B, M, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = np.dot(x, self.W_k).reshape(B, M, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = np.dot(x, self.W_v).reshape(B, M, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) * self.scale
        shift = scores - np.max(scores, axis=-1, keepdims=True)
        attn_weights = np.exp(shift) / np.sum(np.exp(shift), axis=-1, keepdims=True)

        context = np.matmul(attn_weights, V).transpose(0, 2, 1, 3).reshape(B, M, D)
        residual = np.dot(x, self.W_res)
        return np.maximum(0, context + residual)


class AutoIntArchitecture:
    """Full AutoInt neural pipeline with feature field embeddings and stacked interaction layers."""
    def __init__(
        self,
        num_fields: int,
        embed_dim: int = 16,
        num_interaction_layers: int = 3,
        num_heads: int = 4,
        dnn_hidden_dims: List[int] = [64, 32],
        num_classes: int = 1,
    ):
        self.num_fields = num_fields
        self.embed_dim = embed_dim
        self.field_embeddings = np.random.normal(0, 0.1, (num_fields, embed_dim))

        self.attn_layers = [
            MultiHeadTabularSelfAttention(embed_dim, num_heads) for _ in range(num_interaction_layers)
        ]

        # Deep DNN branch
        self.dnn_weights = []
        self.dnn_biases = []
        curr_dim = num_fields * embed_dim
        for h in dnn_hidden_dims:
            self.dnn_weights.append(np.random.normal(0, np.sqrt(2.0 / curr_dim), (curr_dim, h)))
            self.dnn_biases.append(np.zeros(h))
            curr_dim = h

        # Output linear projection head
        total_dim = (num_fields * embed_dim) + dnn_hidden_dims[-1]
        self.head_W = np.random.normal(0, np.sqrt(2.0 / total_dim), (total_dim, num_classes))
        self.head_b = np.zeros(num_classes)

    def forward(self, x: np.ndarray) -> np.ndarray:
        B, M = x.shape
        # Embed fields
        emb = x[:, :, np.newaxis] * self.field_embeddings[np.newaxis, :, :]

        # 1. Attentive Interaction Branch
        h_attn = emb
        for layer in self.attn_layers:
            h_attn = layer.forward(h_attn)
        flat_attn = h_attn.reshape(B, -1)

        # 2. Deep MLP Branch
        flat_raw = emb.reshape(B, -1)
        h_dnn = flat_raw
        for W, b in zip(self.dnn_weights, self.dnn_biases):
            h_dnn = np.maximum(0, np.dot(h_dnn, W) + b)

        # Concatenate branches
        combined = np.concatenate([flat_attn, h_dnn], axis=-1)
        logits = np.dot(combined, self.head_W) + self.head_b
        return logits
