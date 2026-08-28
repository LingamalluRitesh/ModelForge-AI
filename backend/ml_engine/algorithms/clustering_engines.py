"""
ModelForge AI - ML Engine: Unsupervised Clustering & Density Estimation
Implements KMeans++, DBSCAN, and Gaussian Mixture Models (GMM) with Expectation-Maximization.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    """K-Means Clustering with K-Means++ Arthur & Vassilvitskii probabilistic seeding."""

    def __init__(self, n_clusters: int = 5, max_iter: int = 300, tol: float = 1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.cluster_centers_: Optional[np.ndarray] = None
        self.inertia_: float = 0.0

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape

        # KMeans++ initialization
        centers = [X[np.random.randint(n_samples)]]
        for _ in range(1, self.n_clusters):
            dists = np.min(cdist(X, np.array(centers), metric="sqeuclidean"), axis=1)
            probs = dists / np.sum(dists)
            cumprobs = np.cumsum(probs)
            r = np.random.rand()
            next_idx = np.searchsorted(cumprobs, r)
            centers.append(X[next_idx])

        self.cluster_centers_ = np.array(centers)

        for _ in range(self.max_iter):
            # Assign points to closest center
            dists = cdist(X, self.cluster_centers_, metric="sqeuclidean")
            labels = np.argmin(dists, axis=1)

            new_centers = np.array([
                X[labels == k].mean(axis=0) if np.any(labels == k) else self.cluster_centers_[k]
                for k in range(self.n_clusters)
            ])

            center_shift = np.sum((new_centers - self.cluster_centers_) ** 2)
            self.cluster_centers_ = new_centers

            if center_shift < self.tol:
                break

        # Compute inertia
        dists = np.min(cdist(X, self.cluster_centers_, metric="sqeuclidean"), axis=1)
        self.inertia_ = float(np.sum(dists))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        dists = cdist(X, self.cluster_centers_, metric="sqeuclidean")
        return np.argmin(dists, axis=1)


class DBSCAN:
    """Density-Based Spatial Clustering of Applications with Noise (DBSCAN)."""

    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        self.labels_ = np.full(n_samples, -1, dtype=int)  # -1 represents noise

        dist_matrix = cdist(X, X, metric="euclidean")
        visited = np.zeros(n_samples, dtype=bool)
        cluster_id = 0

        for i in range(n_samples):
            if visited[i]:
                continue
            visited[i] = True

            neighbors = np.where(dist_matrix[i] <= self.eps)[0]

            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1  # Noise
            else:
                self.labels_[i] = cluster_id
                queue = list(neighbors)

                while queue:
                    neighbor = queue.pop(0)
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        n_neighbors = np.where(dist_matrix[neighbor] <= self.eps)[0]
                        if len(n_neighbors) >= self.min_samples:
                            queue.extend(n_neighbors)

                    if self.labels_[neighbor] == -1:
                        self.labels_[neighbor] = cluster_id

                cluster_id += 1

        return self
