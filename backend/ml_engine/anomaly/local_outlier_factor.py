"""
ModelForge AI - ML Engine: Local Outlier Factor (LOF)
Implements Breunig, Kriegel, Ng, and Sander Local Outlier Factor for density-based anomaly detection
using k-distance neighborhoods, reachability distance, and local reachability density (lrd).
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.spatial.distance import cdist


class LocalOutlierFactor:
    """Density-based Local Outlier Factor (LOF) anomaly detection algorithm."""

    def __init__(self, n_neighbors: int = 20, contamination: float = 0.05):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.X_fit_: Optional[np.ndarray] = None
        self.k_distances_: Optional[np.ndarray] = None
        self.k_neighbors_: Optional[np.ndarray] = None
        self.lrd_: Optional[np.ndarray] = None
        self.negative_outlier_factor_: Optional[np.ndarray] = None
        self.threshold_: float = -1.5

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        self.X_fit_ = X
        n_samples = X.shape[0]
        k = min(self.n_neighbors, n_samples - 1)

        # Pairwise distance matrix
        dist_matrix = cdist(X, X, metric="euclidean")

        # Sort neighbor distances
        sorted_indices = np.argsort(dist_matrix, axis=1)
        self.k_neighbors_ = sorted_indices[:, 1 : k + 1]  # Exclude self at index 0

        # k-distance of point p (distance to its k-th nearest neighbor)
        self.k_distances_ = np.array([dist_matrix[i, sorted_indices[i, k]] for i in range(n_samples)])

        # Compute Local Reachability Density (lrd)
        self.lrd_ = np.zeros(n_samples)
        for p in range(n_samples):
            neighbors = self.k_neighbors_[p]
            # reachability-distance_k(p, o) = max(k-distance(o), d(p, o))
            reach_dists = np.maximum(self.k_distances_[neighbors], dist_matrix[p, neighbors])
            avg_reach_dist = np.mean(reach_dists)
            self.lrd_[p] = 1.0 / max(1e-10, avg_reach_dist)

        # Compute LOF score: LOF_k(p) = \frac{\sum_{o \in N_k(p)} \frac{lrd(o)}{lrd(p)}}{|N_k(p)|}
        lof_scores = np.zeros(n_samples)
        for p in range(n_samples):
            neighbors = self.k_neighbors_[p]
            lof_scores[p] = np.mean(self.lrd_[neighbors]) / max(1e-10, self.lrd_[p])

        # Scikit-learn convention: negative outlier factor (higher = more normal, lower = anomaly)
        self.negative_outlier_factor_ = -lof_scores
        self.threshold_ = float(np.quantile(self.negative_outlier_factor_, self.contamination))
        return self

    def predict(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Returns 1 for inliers, -1 for anomalies."""
        if X is None or X is self.X_fit_:
            return np.where(self.negative_outlier_factor_ < self.threshold_, -1, 1)

        # Out-of-sample prediction
        X = np.asarray(X, dtype=np.float64)
        n_queries = X.shape[0]
        k = min(self.n_neighbors, len(self.X_fit_) - 1)

        dist_matrix = cdist(X, self.X_fit_, metric="euclidean")
        sorted_indices = np.argsort(dist_matrix, axis=1)

        query_lof = np.zeros(n_queries)
        for q in range(n_queries):
            neighbors = sorted_indices[q, :k]
            reach_dists = np.maximum(self.k_distances_[neighbors], dist_matrix[q, neighbors])
            query_lrd = 1.0 / max(1e-10, np.mean(reach_dists))
            query_lof[q] = np.mean(self.lrd_[neighbors]) / max(1e-10, query_lrd)

        neg_scores = -query_lof
        return np.where(neg_scores < self.threshold_, -1, 1)
