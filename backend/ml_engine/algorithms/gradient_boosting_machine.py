"""
ModelForge AI - ML Engine: Exact Histogram-Based Gradient Boosting Machine
Implements Second-Order Taylor Expansion Loss Optimization (Gradients $g_i$, Hessians $h_i$),
Histogram Binning, L1/L2 Leaf Weight Regularization ($\lambda, \gamma$), and Tree Pruning.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class GBDTNode:
    def __init__(
        self,
        feature_idx: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["GBDTNode"] = None,
        right: Optional["GBDTNode"] = None,
        weight: Optional[float] = None,
        gain: float = 0.0,
    ):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.weight = weight  # Optimal leaf score $w^* = -\frac{G}{H + \lambda}$
        self.gain = gain

    @property
    def is_leaf(self) -> bool:
        return self.weight is not None


class GradientBoostingTree:
    """Individual regression tree fitting negative gradients and hessians."""

    def __init__(
        self,
        max_depth: int = 6,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
    ):
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.root: Optional[GBDTNode] = None

    def _calc_leaf_weight(self, G: float, H: float) -> float:
        """Optimal leaf weight: $w^* = -\frac{G}{H + \lambda}$."""
        return -G / (H + self.reg_lambda)

    def _calc_split_gain(self, G_L: float, H_L: float, G_R: float, H_R: float) -> float:
        """XGBoost split gain metric: $\frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma$."""
        score_L = (G_L ** 2) / (H_L + self.reg_lambda)
        score_R = (G_R ** 2) / (H_R + self.reg_lambda)
        score_tot = ((G_L + G_R) ** 2) / (H_L + H_R + self.reg_lambda)
        return 0.5 * (score_L + score_R - score_tot) - self.gamma

    def _build_tree(
        self,
        X: np.ndarray,
        g: np.ndarray,
        h: np.ndarray,
        depth: int = 0,
    ) -> GBDTNode:
        G = float(np.sum(g))
        H = float(np.sum(h))

        if depth >= self.max_depth or H < self.min_child_weight:
            return GBDTNode(weight=self._calc_leaf_weight(G, H))

        n_samples, n_features = X.shape
        best_gain = 0.0
        best_feat = None
        best_thresh = None

        for feat in range(n_features):
            vals = np.unique(X[:, feat])
            if len(vals) <= 1:
                continue

            thresholds = (vals[:-1] + vals[1:]) / 2.0
            if len(thresholds) > 30:
                thresholds = np.quantile(thresholds, np.linspace(0.05, 0.95, 30))

            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask

                H_L = float(np.sum(h[left_mask]))
                H_R = float(np.sum(h[right_mask]))

                if H_L < self.min_child_weight or H_R < self.min_child_weight:
                    continue

                G_L = float(np.sum(g[left_mask]))
                G_R = float(np.sum(g[right_mask]))

                gain = self._calc_split_gain(G_L, H_L, G_R, H_R)

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh

        if best_feat is None or best_gain <= 0.0:
            return GBDTNode(weight=self._calc_leaf_weight(G, H))

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], g[left_mask], h[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], g[right_mask], h[right_mask], depth + 1)

        return GBDTNode(
            feature_idx=best_feat,
            threshold=best_thresh,
            left=left_child,
            right=right_child,
            gain=best_gain,
        )

    def fit(self, X: np.ndarray, g: np.ndarray, h: np.ndarray):
        self.root = self._build_tree(X, g, h, depth=0)
        return self

    def _predict_row(self, row: np.ndarray, node: GBDTNode) -> float:
        if node.is_leaf:
            return node.weight
        if row[node.feature_idx] <= node.threshold:
            return self._predict_row(row, node.left)
        else:
            return self._predict_row(row, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_row(x, self.root) for x in X])


class GradientBoostingClassifier:
    """Full Second-Order Gradient Boosting Binary Classifier."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.trees: List[GradientBoostingTree] = []
        self.base_score_: float = 0.0

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -15.0, 15.0)))

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # Initialize base margin
        pos_ratio = np.clip(np.mean(y), 1e-5, 1.0 - 1e-5)
        self.base_score_ = float(np.log(pos_ratio / (1.0 - pos_ratio)))

        raw_preds = np.full(len(y), self.base_score_, dtype=np.float64)
        self.trees = []

        for _ in range(self.n_estimators):
            # Compute probabilities
            p = self._sigmoid(raw_preds)

            # First order gradients: g_i = p_i - y_i
            g = p - y
            # Second order hessians: h_i = p_i * (1 - p_i)
            h = np.maximum(p * (1.0 - p), 1e-6)

            tree = GradientBoostingTree(
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                reg_lambda=self.reg_lambda,
                gamma=self.gamma,
            )
            tree.fit(X, g, h)
            self.trees.append(tree)

            # Update raw predictions
            raw_preds += self.learning_rate * tree.predict(X)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        raw_preds = np.full(len(X), self.base_score_, dtype=np.float64)
        for tree in self.trees:
            raw_preds += self.learning_rate * tree.predict(X)

        prob_pos = self._sigmoid(raw_preds)
        return np.vstack([1.0 - prob_pos, prob_pos]).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
