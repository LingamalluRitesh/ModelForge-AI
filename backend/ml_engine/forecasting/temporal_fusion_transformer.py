"""
ModelForge AI - ML Engine: Temporal Fusion Transformer (TFT)
Implements Lim et al. Temporal Fusion Transformers for Interpretable Multi-Horizon Time Series Forecasting
with Variable Selection Networks (VSN), Gated Residual Networks (GRN), and Interpretable Multi-Head Attention.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class GatedResidualNetwork:
    """Nonlinear GRN with Skip Connections and Gated Linear Unit (GLU) suppression."""
    def __init__(self, in_features: int, hidden_dim: int, out_features: int, context_dim: Optional[int] = None):
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_features = out_features
        self.context_dim = context_dim

        std = np.sqrt(2.0 / in_features)
        self.W1 = np.random.normal(0, std, (in_features, hidden_dim))
        self.b1 = np.zeros(hidden_dim)

        self.W2 = np.random.normal(0, np.sqrt(2.0 / hidden_dim), (hidden_dim, out_features * 2))
        self.b2 = np.zeros(out_features * 2)

        # Skip projection
        if in_features != out_features:
            self.W_skip = np.random.normal(0, std, (in_features, out_features))
        else:
            self.W_skip = None

    def _glu(self, x: np.ndarray) -> np.ndarray:
        d = x.shape[-1] // 2
        a = x[..., :d]
        b = x[..., d:]
        sig = 1.0 / (1.0 + np.exp(-np.clip(b, -15.0, 15.0)))
        return a * sig

    def forward(self, x: np.ndarray) -> np.ndarray:
        h1 = np.maximum(0, np.dot(x, self.W1) + self.b1)
        h2 = self._glu(np.dot(h1, self.W2) + self.b2)

        skip = np.dot(x, self.W_skip) if self.W_skip is not None else x
        return (h2 + skip) * np.sqrt(0.5)


class VariableSelectionNetwork:
    """Learns dynamic per-feature importance weights using Softmax-weighted GRN components."""
    def __init__(self, num_inputs: int, in_features: int, hidden_dim: int):
        self.num_inputs = num_inputs
        self.grns = [GatedResidualNetwork(in_features, hidden_dim, hidden_dim) for _ in range(num_inputs)]
        self.selector_grn = GatedResidualNetwork(num_inputs * in_features, hidden_dim, num_inputs)

    def forward(self, features_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        # Concatenate for selection weights
        concat_feat = np.concatenate(features_list, axis=-1)
        weights_raw = self.selector_grn.forward(concat_feat)

        # Softmax over inputs
        shift = weights_raw - np.max(weights_raw, axis=-1, keepdims=True)
        weights = np.exp(shift) / np.sum(np.exp(shift), axis=-1, keepdims=True)

        # Transform individual variables
        transformed = [self.grns[i].forward(features_list[i]) for i in range(self.num_inputs)]
        stacked = np.stack(transformed, axis=-2)  # (..., num_inputs, hidden_dim)

        # Weighted combination
        combined = np.sum(stacked * weights[..., np.newaxis], axis=-2)
        return combined, weights
