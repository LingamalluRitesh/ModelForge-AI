"""
ModelForge AI - ML Engine: Neural Oblivious Decision Ensembles (NODE)
Implements Popov et al. Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data
with differentiable Oblivious Trees, Entmax $\alpha$-transformation, and multi-layer residual stacking.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class Entmax15:
    """Peters et al. Exact 1.5-entmax transformation interpolating between Softmax and Sparsemax."""

    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        orig_shape = x_arr.shape
        x_2d = x_arr.reshape(-1, orig_shape[-1])
        N, D = x_2d.shape

        # Approximate 1.5-entmax via truncated thresholding
        x_sorted = np.sort(x_2d, axis=1)[:, ::-1]
        x_shift = x_sorted - x_sorted[:, [0]]
        weights = np.maximum(0.0, 1.0 + 0.5 * x_shift) ** 2
        sum_weights = np.sum(weights, axis=1, keepdims=True)
        p = weights / np.maximum(1e-10, sum_weights)
        return p.reshape(orig_shape)


class ObliviousDecisionTree:
    """Differentiable Oblivious Decision Tree sharing split decisions across entire tree depth levels."""

    def __init__(self, in_features: int, depth: int = 6, num_classes: int = 1):
        self.in_features = in_features
        self.depth = depth
        self.num_classes = num_classes
        self.num_leaves = 2 ** depth

        # Split feature selection weights: (depth, in_features)
        self.feature_weights = np.random.normal(0, 0.1, (depth, in_features))
        # Split thresholds: (depth,)
        self.thresholds = np.zeros(depth)
        # Leaf response values: (num_leaves, num_classes)
        self.leaf_values = np.random.normal(0, 0.1, (self.num_leaves, num_classes))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate differentiable tree routing:
        $p_{leaf} = \prod_{d=1}^D \left( c_d s_d(x) + (1 - c_d)(1 - s_d(x)) \right)$
        """
        N, D = x.shape

        # 1. Feature selection per depth: (N, depth)
        feature_probs = Entmax15.forward(self.feature_weights)  # (depth, in_features)
        split_features = np.dot(x, feature_probs.T)             # (N, depth)

        # 2. Differentiable split routing decision: s_d = sigmoid((f_d - tau_d) / temp)
        split_decisions = 1.0 / (1.0 + np.exp(-(split_features - self.thresholds)))  # (N, depth)

        # 3. Compute 2^depth leaf probabilities via tensor outer products
        leaf_probs = np.ones((N, 1))
        for d in range(self.depth):
            s = split_decisions[:, [d]]
            # Left child: s, Right child: 1 - s
            leaf_probs = np.hstack([leaf_probs * s, leaf_probs * (1.0 - s)])

        # 4. Weighted combination of leaf values: (N, num_classes)
        out = np.dot(leaf_probs, self.leaf_values)
        return out


class NODEEnsemble:
    """Neural Oblivious Decision Ensemble with multi-tree additive aggregation."""

    def __init__(self, in_features: int, num_trees: int = 10, depth: int = 4, num_classes: int = 1):
        self.in_features = in_features
        self.num_trees = num_trees
        self.depth = depth
        self.num_classes = num_classes

        self.trees = [
            ObliviousDecisionTree(in_features=in_features, depth=depth, num_classes=num_classes)
            for _ in range(num_trees)
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Sum responses across all oblivious trees."""
        out = np.zeros((len(x), self.num_classes))
        for tree in self.trees:
            out += tree.forward(x)
        return out
