"""
ModelForge AI - Forecasting Engine: Temporal Fusion Transformer (TFT) Architecture
Implements Lim et al. Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
with Variable Selection Networks (VSN), Gated Residual Networks (GRN), and Interpretable Multi-Head Self-Attention.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class GatedResidualNetwork:
    """GRN block with non-linear processing, optional context conditioning, and GLU gating."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, context_dim: Optional[int] = None):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.has_context = context_dim is not None

        std = np.sqrt(2.0 / in_dim)
        self.W1 = np.random.normal(0, std, (in_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)

        if self.has_context and context_dim:
            self.W_c = np.random.normal(0, np.sqrt(2.0 / context_dim), (context_dim, hidden_dim))

        self.W2 = np.random.normal(0, np.sqrt(2.0 / hidden_dim), (hidden_dim, out_dim * 2))
        self.b2 = np.zeros(out_dim * 2)

        # Residual skip projection if dimensions differ
        self.W_res = np.random.normal(0, std, (in_dim, out_dim)) if in_dim != out_dim else None

    def forward(self, x: np.ndarray, context: Optional[np.ndarray] = None) -> np.ndarray:
        # First dense layer
        h = np.dot(x, self.W1) + self.b1
        if self.has_context and context is not None:
            h += np.dot(context, self.W_c)
        h = np.maximum(0, h)  # ELU/ReLU activation

        # Second dense layer into GLU
        glu_in = np.dot(h, self.W2) + self.b2
        a = glu_in[..., : self.out_dim]
        b = glu_in[..., self.out_dim :]
        glu_out = a * (1.0 / (1.0 + np.exp(-b)))

        # Residual skip
        skip = np.dot(x, self.W_res) if self.W_res is not None else x
        return skip + glu_out


class VariableSelectionNetwork:
    """VSN dynamically weights relevant continuous and categorical variables per time step."""

    def __init__(self, num_inputs: int, input_dim: int, hidden_dim: int, context_dim: Optional[int] = None):
        self.num_inputs = num_inputs
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Flattened GRN for computing Softmax feature importance weights
        self.weight_grn = GatedResidualNetwork(num_inputs * input_dim, hidden_dim, num_inputs, context_dim)
        # Per-feature processing GRNs
        self.feature_grns = [
            GatedResidualNetwork(input_dim, hidden_dim, hidden_dim, context_dim) for _ in range(num_inputs)
        ]

    def forward(self, inputs: List[np.ndarray], context: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        inputs: list of num_inputs tensors each (B, T, input_dim)
        returns: (selected_representation, variable_weights)
        """
        flat_inputs = np.concatenate(inputs, axis=-1)
        raw_weights = self.weight_grn.forward(flat_inputs, context)
        # Softmax along variable dimension
        shift = raw_weights - np.max(raw_weights, axis=-1, keepdims=True)
        var_weights = np.exp(shift) / np.sum(np.exp(shift), axis=-1, keepdims=True)  # (B, T, num_inputs)

        # Process each feature through its own GRN
        processed_features = [
            self.feature_grns[i].forward(inputs[i], context) for i in range(self.num_inputs)
        ]
        stacked = np.stack(processed_features, axis=-2)  # (B, T, num_inputs, hidden_dim)

        # Weighted combination
        weights_expanded = var_weights[..., np.newaxis]
        selected = np.sum(weights_expanded * stacked, axis=-2)  # (B, T, hidden_dim)

        return selected, var_weights
