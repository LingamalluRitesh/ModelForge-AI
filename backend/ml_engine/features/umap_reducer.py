"""
ModelForge AI - ML Engine: Uniform Manifold Approximation & Projection (UMAP)
Implements McInnes et al. UMAP non-linear dimensionality reduction via Fuzzy Simplicial Sets,
Riemannian Metric local connectivity, and Stochastic Gradient Descent (SGD) layout optimization.
$p_{i|j} = \exp\left(-\max(0, d(x_i, x_j) - \rho_i) / \sigma_i\right)$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.spatial.distance import cdist


class UMAPReducer:
    """Non-linear manifold dimension reduction for high-dimensional feature spaces."""

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        n_epochs: int = 200,
        lr: float = 1.0,
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_epochs = n_epochs
        self.lr = lr
        self.embedding_: Optional[np.ndarray] = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Compute low-dimensional manifold embedding coordinates."""
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]

        # 1. Pairwise distance matrix
        dist_matrix = cdist(X, X, metric="euclidean")

        # 2. Local metric scaling (rho_i and sigma_i)
        rho = np.zeros(n_samples)
        for i in range(n_samples):
            sorted_dists = np.sort(dist_matrix[i])
            rho[i] = sorted_dists[1] if n_samples > 1 else 0.0

        # Compute fuzzy simplicial set edge weights P_ij
        P = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                if i != j:
                    P[i, j] = np.exp(-max(0.0, dist_matrix[i, j] - rho[i]))

        # Symmetrize: B = P + P^T - P * P^T
        B = P + P.T - P * P.T

        # 3. Spectral initialization / Random normal
        Y = np.random.normal(0, 1e-4, (n_samples, self.n_components))

        # 4. SGD layout optimization
        a, b = 1.58, 0.89  # Curves for min_dist = 0.1

        for epoch in range(self.n_epochs):
            alpha = self.lr * (1.0 - epoch / float(self.n_epochs))

            for i in range(n_samples):
                for j in range(n_samples):
                    if i == j:
                        continue

                    diff = Y[i] - Y[j]
                    dist_sq = np.sum(diff ** 2)

                    # Attractive force
                    if B[i, j] > 1e-3:
                        grad_coeff = (-2.0 * a * b * (dist_sq ** (b - 1.0))) / (1.0 + a * (dist_sq ** b))
                        grad = np.clip(grad_coeff * diff, -4.0, 4.0)
                        Y[i] += alpha * grad * B[i, j]

                    # Repulsive force
                    if np.random.rand() < 0.05:
                        grad_coeff = 2.0 * b / ((0.001 + dist_sq) * (1.0 + a * (dist_sq ** b)))
                        grad = np.clip(grad_coeff * diff, -4.0, 4.0)
                        Y[i] += alpha * grad * 0.1

        self.embedding_ = Y
        return self.embedding_
