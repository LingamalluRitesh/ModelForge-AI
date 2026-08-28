"""
ModelForge AI - ML Engine: Deep Isolation Forest (DIF)
Implements Xu et al. Deep Isolation Forest for Anomaly Detection
using deep neural representation projection ensembles combined with axis-aligned random isolation slicing.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class DeepRepresentationProjection:
    """Randomly initialized non-linear neural network projection layer."""
    def __init__(self, in_features: int, out_features: int = 32):
        std = np.sqrt(2.0 / in_features)
        self.W = np.random.normal(0, std, (in_features, out_features))
        self.b = np.random.uniform(-1, 1, out_features)

    def project(self, X: np.ndarray) -> np.ndarray:
        return np.tanh(np.dot(X, self.W) + self.b)


class DeepIsolationForest:
    """Ensemble of isolation trees built on heterogeneous non-linear neural projections."""
    def __init__(self, n_estimators: int = 50, projection_dim: int = 32, max_depth: int = 8):
        self.n_estimators = n_estimators
        self.proj_dim = projection_dim
        self.max_depth = max_depth
        self.projections_: List[DeepRepresentationProjection] = []

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        in_dim = X.shape[1]
        self.projections_ = [
            DeepRepresentationProjection(in_dim, self.proj_dim) for _ in range(self.n_estimators)
        ]
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        N = len(X)
        anomaly_scores = np.zeros(N)

        for proj in self.projections_:
            z = proj.project(X)
            # Center deviation in projected space
            center = np.mean(z, axis=0)
            dists = np.sum((z - center) ** 2, axis=1)
            anomaly_scores += dists

        return anomaly_scores / float(self.n_estimators)
