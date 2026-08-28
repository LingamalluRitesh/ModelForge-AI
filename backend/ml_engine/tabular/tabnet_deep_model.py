"""
ModelForge AI - ML Engine: TabNet Attentive Interpretable Tabular Learning
Implements Arik & Pfister TabNet: Attentive Interpretable Tabular Learning with
Sequential Multi-Step Attention, Sparsemax / Entmax sparsity selection, Prior Scales $\gamma$,
and Feature Transformer blocks with Ghost Batch Normalization.
$M[i] = \text{sparsemax}(P[i-1] \cdot h_i(a[i-1]))$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class Sparsemax:
    """Martins & Astudillo Sparsemax activation returning exact sparse probabilities."""

    @staticmethod
    def forward(z: np.ndarray) -> np.ndarray:
        """
        Projects $z$ onto the probability simplex: $\arg\min_p ||p - z||_2^2 \text{ s.t. } p \in \Delta$.
        """
        z_arr = np.asarray(z, dtype=np.float64)
        orig_shape = z_arr.shape
        z_2d = z_arr.reshape(-1, orig_shape[-1])
        N, D = z_2d.shape

        # 1. Sort z descending
        z_sorted = np.sort(z_2d, axis=1)[:, ::-1]

        # 2. Find threshold tau(z)
        z_cumsum = np.cumsum(z_sorted, axis=1)
        k_indices = np.arange(1, D + 1)
        support = (1.0 + k_indices * z_sorted) > z_cumsum

        # Number of active coordinates
        k_z = np.sum(support, axis=1)
        tau_z = (z_cumsum[np.arange(N), k_z - 1] - 1.0) / k_z

        # 3. Output max(z - tau, 0)
        p = np.maximum(0.0, z_2d - tau_z[:, np.newaxis])
        return p.reshape(orig_shape)


class GhostBatchNorm:
    """Virtual Ghost Batch Normalization for large-batch tabular stability."""

    def __init__(self, num_features: int, virtual_batch_size: int = 128, momentum: float = 0.02):
        self.num_features = num_features
        self.vbs = virtual_batch_size
        self.momentum = momentum

        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        N, D = x.shape
        if not training or N <= self.vbs:
            mean = self.running_mean if not training else np.mean(x, axis=0)
            var = self.running_var if not training else np.var(x, axis=0)
            return self.gamma * ((x - mean) / np.sqrt(var + 1e-5)) + self.beta

        # Split into virtual chunks
        chunks = [x[i : i + self.vbs] for i in range(0, N, self.vbs)]
        out_chunks = []
        for ch in chunks:
            ch_mean = np.mean(ch, axis=0)
            ch_var = np.var(ch, axis=0)
            self.running_mean = (1.0 - self.momentum) * self.running_mean + self.momentum * ch_mean
            self.running_var = (1.0 - self.momentum) * self.running_var + self.momentum * ch_var
            norm_ch = self.gamma * ((ch - ch_mean) / np.sqrt(ch_var + 1e-5)) + self.beta
            out_chunks.append(norm_ch)

        return np.vstack(out_chunks)


class FeatureTransformer:
    """Shared and decision step-dependent Gated Linear Unit (GLU) blocks."""

    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features

        std = np.sqrt(2.0 / in_features)
        self.W1 = np.random.normal(0, std, (in_features, out_features * 2))
        self.b1 = np.zeros(out_features * 2)
        self.bn1 = GhostBatchNorm(out_features * 2)

        self.W2 = np.random.normal(0, np.sqrt(2.0 / out_features), (out_features, out_features * 2))
        self.b2 = np.zeros(out_features * 2)
        self.bn2 = GhostBatchNorm(out_features * 2)

    def _glu(self, x: np.ndarray) -> np.ndarray:
        """Gated Linear Unit: $GLU(a, b) = a \otimes \sigma(b)$."""
        d = x.shape[-1] // 2
        a = x[:, :d]
        b = x[:, d:]
        sig = 1.0 / (1.0 + np.exp(-np.clip(b, -15.0, 15.0)))
        return a * sig

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        h1 = self._glu(self.bn1.forward(np.dot(x, self.W1) + self.b1, training))
        h2 = self._glu(self.bn2.forward(np.dot(h1, self.W2) + self.b2, training))
        return (h1 + h2) * np.sqrt(0.5)


class TabNetModel:
    """TabNet Sequential Decision Architecture."""

    def __init__(
        self,
        in_features: int,
        out_features: int = 2,
        n_d: int = 16,
        n_a: int = 16,
        n_steps: int = 4,
        gamma: float = 1.3,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma

        # Attentive transformers per step
        self.att_transformers = [
            FeatureTransformer(n_a, in_features) for _ in range(n_steps)
        ]
        # Feature transformers per step
        self.feat_transformers = [
            FeatureTransformer(in_features, n_d + n_a) for _ in range(n_steps)
        ]

        # Final classification head
        self.head_W = np.random.normal(0, 0.05, (n_d, out_features))
        self.head_b = np.zeros(out_features)

    def forward(self, x: np.ndarray, training: bool = True) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Forward TabNet evaluation returning predictions and per-step sparse attention masks $M[i]$.
        """
        N, D = x.shape
        prior_scales = np.ones((N, D))
        out_accumulator = np.zeros((N, self.n_d))
        step_masks: List[np.ndarray] = []

        # Initial attentive representation
        a = np.zeros((N, self.n_a))

        for step in range(self.n_steps):
            # 1. Attentive Transformer -> Compute Mask M[i]
            att_features = self.att_transformers[step].forward(a, training)
            mask_logits = prior_scales * att_features
            M = Sparsemax.forward(mask_logits)
            step_masks.append(M)

            # Update prior scales: P[i] = P[i-1] * (gamma - M[i])
            prior_scales = prior_scales * (self.gamma - M)

            # 2. Feature Transformer on masked input
            x_masked = x * M
            step_features = self.feat_transformers[step].forward(x_masked, training)

            # Split into decision features (d) and attentive features (a)
            d = step_features[:, : self.n_d]
            a = step_features[:, self.n_d :]

            # Accumulate decision output with ReLU
            out_accumulator += np.maximum(0, d)

        # Final linear prediction
        logits = np.dot(out_accumulator, self.head_W) + self.head_b
        return logits, step_masks
