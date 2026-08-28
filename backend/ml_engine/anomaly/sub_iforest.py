"""
ModelForge AI - ML Engine: Subspace Isolation Forest (Sub-iForest)
Implements Hariri et al. Extended Isolation Forest with Random Linear Cutting Hyperplanes
eliminating axis-aligned slicing artifacts and identifying anomalies in arbitrary oblique subspace geometries.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class SubspaceIsolationTree:
    """Random linear hyper-plane cutting binary isolation tree."""
    def __init__(self, max_depth: int = 10):
        self.max_depth = max_depth
        self.normal_vector: Optional[np.ndarray] = None
        self.intercept: Optional[float] = None
        self.left: Optional["SubspaceIsolationTree"] = None
        self.right: Optional["SubspaceIsolationTree"] = None
        self.size = 0

    def fit(self, X: np.ndarray, depth: int = 0):
        N, D = X.shape
        self.size = N
        if depth >= self.max_depth or N <= 1:
            return self

        # Sample random normal hyperplane vector on unit sphere
        n = np.random.normal(0, 1, D)
        n /= max(1e-8, np.linalg.norm(n))
        self.normal_vector = n

        # Project data onto normal vector
        projections = np.dot(X, n)
        p_min, p_max = np.min(projections), np.max(projections)

        if p_min == p_max:
            return self

        # Random cutting point
        self.intercept = np.random.uniform(p_min, p_max)
        left_mask = projections < self.intercept
        right_mask = ~left_mask

        if np.sum(left_mask) > 0 and np.sum(right_mask) > 0:
            self.left = SubspaceIsolationTree(self.max_depth).fit(X[left_mask], depth + 1)
            self.right = SubspaceIsolationTree(self.max_depth).fit(X[right_mask], depth + 1)

        return self

    def path_length(self, x: np.ndarray, current_depth: int = 0) -> float:
        if self.left is None or self.right is None:
            return float(current_depth)

        proj = float(np.dot(x, self.normal_vector))
        if proj < self.intercept:
            return self.left.path_length(x, current_depth + 1)
        else:
            return self.right.path_length(x, current_depth + 1)
