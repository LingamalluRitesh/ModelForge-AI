"""
ModelForge AI - ML Engine: Extended Isolation Forest (EIF)
Implements Hariri, Kind, and Brunner Extended Isolation Forest for high-dimensional anomaly detection
using random cutting hyperplanes with generalized normal vectors rather than axis-aligned cuts.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class EIFNode:
    def __init__(
        self,
        left: Optional["EIFNode"] = None,
        right: Optional["EIFNode"] = None,
        n: Optional[np.ndarray] = None,  # Normal vector of separating hyperplane
        p: Optional[np.ndarray] = None,  # Intercept point on hyperplane
        size: int = 0,
    ):
        self.left = left
        self.right = right
        self.n = n
        self.p = p
        self.size = size

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class ExtendedIsolationTree:
    """Individual Isolation Tree branching via random slope hyperplanes."""

    def __init__(self, max_depth: int, extension_level: int = 0):
        self.max_depth = max_depth
        self.extension_level = extension_level  # 0 = standard IF, >0 = extended dimensionality
        self.root: Optional[EIFNode] = None

    def fit(self, X: np.ndarray, depth: int = 0) -> EIFNode:
        n_samples, n_dims = X.shape

        if depth >= self.max_depth or n_samples <= 1:
            return EIFNode(size=n_samples)

        # Select random slope hyperplane
        if self.extension_level == 0:
            # Axis-aligned
            feat = np.random.randint(n_dims)
            n = np.zeros(n_dims)
            n[feat] = 1.0
        else:
            # Rotated hyperplane
            n = np.random.normal(0, 1, n_dims)
            # Mask out non-participating dimensions if extension_level < n_dims - 1
            if self.extension_level < n_dims - 1:
                mask = np.random.choice(n_dims, n_dims - 1 - self.extension_level, replace=False)
                n[mask] = 0.0
            n = n / max(1e-6, np.linalg.norm(n))

        # Select random intercept point p between min and max projections
        projections = np.dot(X, n)
        min_p = np.min(projections)
        max_p = np.max(projections)

        if min_p == max_p:
            return EIFNode(size=n_samples)

        p_val = np.random.uniform(min_p, max_p)

        left_mask = projections < p_val
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return EIFNode(size=n_samples)

        left_node = self.fit(X[left_mask], depth + 1)
        right_node = self.fit(X[right_mask], depth + 1)

        return EIFNode(left=left_node, right=right_node, n=n, p=p_val, size=n_samples)


class ExtendedIsolationForest:
    """Ensemble of Extended Isolation Trees producing anomaly score $s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: int = 256,
        extension_level: int = 1,
        contamination: float = 0.05,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.extension_level = extension_level
        self.contamination = contamination
        self.trees: List[EIFNode] = []
        self.threshold_: float = 0.5

    def _c(self, n: int) -> float:
        """Average path length of unsuccessful searches in Binary Search Tree (BST)."""
        if n <= 1:
            return 1.0
        if n == 2:
            return 1.0
        euler_mascheroni = 0.5772156649
        return 2.0 * (np.log(n - 1) + euler_mascheroni) - (2.0 * (n - 1.0) / n)

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        subsample_size = min(n_samples, self.max_samples)
        max_depth = int(np.ceil(np.log2(max(1, subsample_size))))

        self.trees = []
        for _ in range(self.n_estimators):
            idx = np.random.choice(n_samples, subsample_size, replace=False)
            tree = ExtendedIsolationTree(max_depth=max_depth, extension_level=self.extension_level)
            root = tree.fit(X[idx])
            self.trees.append(root)

        # Determine anomaly score threshold based on contamination percentile
        scores = self.score_samples(X)
        self.threshold_ = float(np.quantile(scores, 1.0 - self.contamination))
        return self

    def _path_length(self, x: np.ndarray, node: EIFNode, curr_depth: int = 0) -> float:
        if node.is_leaf:
            return curr_depth + self._c(node.size)

        projection = float(np.dot(x, node.n))
        if projection < node.p:
            return self._path_length(x, node.left, curr_depth + 1)
        else:
            return self._path_length(x, node.right, curr_depth + 1)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly score where values close to 1 are anomalies."""
        X = np.asarray(X, dtype=np.float64)
        c_factor = self._c(self.max_samples)

        path_lengths = np.zeros(len(X))
        for tree in self.trees:
            for i, x in enumerate(X):
                path_lengths[i] += self._path_length(x, tree)

        avg_path = path_lengths / len(self.trees)
        return 2.0 ** (-avg_path / c_factor)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns 1 for inliers and -1 for anomalies."""
        scores = self.score_samples(X)
        return np.where(scores >= self.threshold_, -1, 1)
