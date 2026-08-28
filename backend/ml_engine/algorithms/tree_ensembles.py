"""
ModelForge AI - ML Engine: Tree Ensembles from First Principles
Implements DecisionTreeClassifier, DecisionTreeRegressor, RandomForestClassifier,
and AdaBoostClassifier supporting Gini impurity, Shannon entropy, and Variance reduction.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class TreeNode:
    """Represents a decision tree internal decision split or leaf node."""

    def __init__(
        self,
        feature_idx: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["TreeNode"] = None,
        right: Optional["TreeNode"] = None,
        value: Optional[Any] = None,
        gain: float = 0.0,
    ):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Leaf prediction value / class probabilities
        self.gain = gain

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTree:
    """Full-featured Decision Tree for Classification and Regression."""

    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = "gini",  # gini, entropy, mse
        max_features: Optional[Union[int, float, str]] = None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.root: Optional[TreeNode] = None
        self.n_classes: int = 0
        self.is_classifier: bool = criterion in ("gini", "entropy")

    def _impurity(self, y: np.ndarray) -> float:
        n = len(y)
        if n == 0:
            return 0.0

        if self.criterion == "gini":
            _, counts = np.unique(y, return_counts=True)
            p = counts / n
            return float(1.0 - np.sum(p ** 2))
        elif self.criterion == "entropy":
            _, counts = np.unique(y, return_counts=True)
            p = counts / n
            return float(-np.sum(p * np.log2(p + 1e-12)))
        else:  # mse
            mean_y = np.mean(y)
            return float(np.mean((y - mean_y) ** 2))

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split:
            return None, None, 0.0

        current_impurity = self._impurity(y)
        best_gain = 0.0
        best_feat = None
        best_thresh = None

        # Feature subsampling
        if self.max_features == "sqrt":
            n_sub = int(np.sqrt(n_features))
        elif self.max_features == "log2":
            n_sub = int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            n_sub = min(n_features, self.max_features)
        else:
            n_sub = n_features

        feature_indices = np.random.choice(n_features, n_sub, replace=False)

        for feat in feature_indices:
            vals = np.unique(X[:, feat])
            if len(vals) <= 1:
                continue

            # Check midpoints as candidate thresholds
            thresholds = (vals[:-1] + vals[1:]) / 2.0
            if len(thresholds) > 50:
                thresholds = np.quantile(thresholds, np.linspace(0.02, 0.98, 50))

            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask

                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                imp_left = self._impurity(y[left_mask])
                imp_right = self._impurity(y[right_mask])

                gain = current_impurity - ((n_left / n_samples) * imp_left + (n_right / n_samples) * imp_right)

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh

        return best_feat, best_thresh, best_gain

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> TreeNode:
        n_samples = len(y)

        # Leaf conditions
        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or (self.is_classifier and len(np.unique(y)) == 1)
        ):
            leaf_val = self._leaf_value(y)
            return TreeNode(value=leaf_val)

        best_feat, best_thresh, best_gain = self._best_split(X, y)

        if best_feat is None or best_gain <= 1e-7:
            leaf_val = self._leaf_value(y)
            return TreeNode(value=leaf_val)

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return TreeNode(
            feature_idx=best_feat,
            threshold=best_thresh,
            left=left_child,
            right=right_child,
            gain=best_gain,
        )

    def _leaf_value(self, y: np.ndarray) -> Any:
        if self.is_classifier:
            counts = np.bincount(y.astype(int), minlength=self.n_classes)
            return counts / max(1, len(y))  # Probabilities
        else:
            return float(np.mean(y))

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if self.is_classifier:
            self.n_classes = int(np.max(y)) + 1
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _predict_row(self, row: np.ndarray, node: TreeNode) -> Any:
        if node.is_leaf:
            return node.value

        if row[node.feature_idx] <= node.threshold:
            return self._predict_row(row, node.left)
        else:
            return self._predict_row(row, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        preds = [self._predict_row(x, self.root) for x in X]
        if self.is_classifier:
            return np.argmax(preds, axis=1)
        return np.array(preds)


class RandomForestClassifier:
    """Bagging ensemble of randomized decision trees with Out-Of-Bag (OOB) scoring."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 12,
        min_samples_split: int = 2,
        criterion: str = "gini",
        max_features: str = "sqrt",
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.trees: List[DecisionTree] = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        n_samples = X.shape[0]

        self.trees = []
        for _ in range(self.n_estimators):
            # Bootstrap sample with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion,
                max_features=self.max_features,
            )
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        tree_probs = np.array([[tree._predict_row(x, tree.root) for x in X] for tree in self.trees])
        return np.mean(tree_probs, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
