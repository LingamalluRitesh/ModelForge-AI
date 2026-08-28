"""
ModelForge AI - ML Engine: Ghost Batch Normalization (GBN)
Implements Hoffer et al. Train longer, generalize better: closing the generalization gap with large batch sizes
partitioning large mini-batches into virtual sub-batches to reduce generalization error.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class GhostBatchNorm:
    """Partitions batch $B$ into virtual sub-batches of size $V$ to compute independent normalization statistics."""
    def __init__(self, num_features: int, virtual_batch_size: int = 32, momentum: float = 0.1, eps: float = 1e-5):
        self.virtual_bs = virtual_batch_size
        self.momentum = momentum
        self.eps = eps
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, X: np.ndarray, is_training: bool = True) -> np.ndarray:
        if not is_training:
            norm = (X - self.running_mean) / np.sqrt(self.running_var + self.eps)
            return self.gamma * norm + self.beta

        N = len(X)
        out_chunks = []
        for i in range(0, N, self.virtual_bs):
            chunk = X[i : i + self.virtual_bs]
            mean = np.mean(chunk, axis=0)
            var = np.var(chunk, axis=0)

            # Update running stats
            self.running_mean = (1.0 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1.0 - self.momentum) * self.running_var + self.momentum * var

            chunk_norm = (chunk - mean) / np.sqrt(var + self.eps)
            out_chunks.append(self.gamma * chunk_norm + self.beta)

        return np.vstack(out_chunks)
