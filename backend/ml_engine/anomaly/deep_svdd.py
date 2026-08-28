"""
ModelForge AI - ML Engine: Deep Support Vector Data Description (Deep SVDD)
Implements Ruff et al. Deep One-Class Classification
training neural networks to map normal data into a minimal hypersphere centered at point $c$.
$\min_{\mathcal{W}} rac{1}{n} \sum_{i=1}^n ||\phi(x_i; \mathcal{W}) - c||^2 + rac{\lambda}{2} \sum_{l=1}^L ||W^l||_F^2$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class DeepSVDDAnomalyDetector:
    """Deep one-class neural network anomaly detector."""
    def __init__(self, in_features: int, latent_dim: int = 16, nu: float = 0.05):
        self.in_features = in_features
        self.latent_dim = latent_dim
        self.nu = nu

        std = np.sqrt(2.0 / in_features)
        self.W1 = np.random.normal(0, std, (in_features, 64))
        self.b1 = np.zeros(64)
        self.W2 = np.random.normal(0, np.sqrt(2.0 / 64), (64, latent_dim))
        self.b2 = np.zeros(latent_dim)

        self.c_: Optional[np.ndarray] = None
        self.R2_: float = 0.0

    def fit(self, X: np.ndarray, epochs: int = 20, lr: float = 0.01):
        X = np.asarray(X, dtype=float)
        N = len(X)

        # 1. Forward pass to initialize center c
        h1 = np.maximum(0, np.dot(X, self.W1) + self.b1)
        z = np.dot(h1, self.W2) + self.b2
        self.c_ = np.mean(z, axis=0)

        # 2. Representation refinement
        for _ in range(epochs):
            h1 = np.maximum(0, np.dot(X, self.W1) + self.b1)
            z = np.dot(h1, self.W2) + self.b2
            diff = z - self.c_  # (N, latent_dim)

            # Gradient update
            grad_z = (2.0 / N) * diff
            grad_W2 = np.dot(h1.T, grad_z)
            self.W2 -= lr * grad_W2

        # Compute radius R^2 as (1 - nu) quantile of distances
        dists = np.sum((z - self.c_) ** 2, axis=1)
        self.R2_ = float(np.quantile(dists, 1.0 - self.nu))
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        h1 = np.maximum(0, np.dot(X, self.W1) + self.b1)
        z = np.dot(h1, self.W2) + self.b2
        dists = np.sum((z - self.c_) ** 2, axis=1)
        return dists - self.R2_
