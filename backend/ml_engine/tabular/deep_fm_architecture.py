"""
ModelForge AI - ML Engine: DeepFM Architecture
Implements Guo et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction
combining 1st-order linear features, 2nd-order FM inner product interactions, and deep high-order DNN representations.
$y = 	ext{sigmoid}\left( w_0 + \sum_{i=1}^d w_i x_i + \sum_{i=1}^d \sum_{j=i+1}^d \langle v_i, v_j angle x_i x_j + y_{DNN} ight)$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class DeepFMModel:
    """Deep Factorization Machine combining shallow explicit 2nd-order crosses and deep implicit representations."""
    def __init__(self, num_fields: int, embed_dim: int = 16, hidden_dims: List[int] = [128, 64], num_classes: int = 1):
        self.num_fields = num_fields
        self.embed_dim = embed_dim

        # 1st-order linear weights
        self.w_linear = np.random.normal(0, 0.05, (num_fields, 1))
        self.b_linear = 0.0

        # 2nd-order FM embeddings: (num_fields, embed_dim)
        self.V = np.random.normal(0, np.sqrt(2.0 / embed_dim), (num_fields, embed_dim))

        # Deep DNN weights
        self.dnn_weights = []
        self.dnn_biases = []
        curr_dim = num_fields * embed_dim
        for h_dim in hidden_dims:
            self.dnn_weights.append(np.random.normal(0, np.sqrt(2.0 / curr_dim), (curr_dim, h_dim)))
            self.dnn_biases.append(np.zeros(h_dim))
            curr_dim = h_dim

        self.dnn_head_W = np.random.normal(0, np.sqrt(2.0 / curr_dim), (curr_dim, num_classes))
        self.dnn_head_b = np.zeros(num_classes)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: (B, num_fields) continuous/dense feature inputs
        """
        B, M = x.shape

        # 1. First-Order Linear Component: sum_i w_i x_i
        linear_part = np.dot(x, self.w_linear) + self.b_linear  # (B, 1)

        # 2. Second-Order FM Interaction Component:
        # $1/2 \sum_f [(\sum_i v_{i,f} x_i)^2 - \sum_i v_{i,f}^2 x_i^2]$
        xv = x[:, :, np.newaxis] * self.V[np.newaxis, :, :]  # (B, M, D)
        sum_xv = np.sum(xv, axis=1)                          # (B, D)
        sum_xv_sq = sum_xv ** 2                              # (B, D)

        sq_xv = xv ** 2                                      # (B, M, D)
        sq_sum_xv = np.sum(sq_xv, axis=1)                    # (B, D)

        fm_part = 0.5 * np.sum(sum_xv_sq - sq_sum_xv, axis=1, keepdims=True)  # (B, 1)

        # 3. Deep DNN Component
        flat_emb = xv.reshape(B, -1)
        h = flat_emb
        for W, b in zip(self.dnn_weights, self.dnn_biases):
            h = np.maximum(0, np.dot(h, W) + b)
        dnn_part = np.dot(h, self.dnn_head_W) + self.dnn_head_b  # (B, 1)

        # Sum components
        logits = linear_part + fm_part + dnn_part
        return logits
