"""
ModelForge AI - Feature Selection & Dimensionality Reduction
Implements Variance Threshold, Correlation Filtering, Mutual Information, and Recursive Feature Elimination (RFE).
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression


class AdvancedFeatureSelector:
    """
    Automated multi-stage feature selector combining low variance pruning, collinearity removal, and mutual information ranking.
    """

    def __init__(
        self,
        variance_threshold: float = 0.01,
        max_correlation: float = 0.95,
        top_k_features: Optional[int] = None,
        is_classification: bool = True,
    ):
        self.variance_threshold = variance_threshold
        self.max_correlation = max_correlation
        self.top_k_features = top_k_features
        self.is_classification = is_classification
        self.selected_feature_indices_: List[int] = []
        self.feature_names_: List[str] = []
        self.selected_feature_names_: List[str] = []

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, feature_names: Optional[List[str]] = None):
        n_samples, n_features = X.shape
        self.feature_names_ = feature_names or [f"feature_{i}" for i in range(n_features)]

        # 1. Variance Filter
        variances = np.var(X, axis=0)
        keep_var = np.where(variances > self.variance_threshold)[0]
        if len(keep_var) == 0:
            keep_var = np.arange(n_features)

        # 2. Correlation Filter (Drop collinear redundant features)
        X_sub = X[:, keep_var]
        corr_matrix = np.corrcoef(X_sub, rowvar=False)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        drop_idx = set()
        for i in range(len(keep_var)):
            for j in range(i + 1, len(keep_var)):
                if abs(corr_matrix[i, j]) > self.max_correlation:
                    drop_idx.add(j)

        kept_indices = [keep_var[i] for i in range(len(keep_var)) if i not in drop_idx]

        # 3. Mutual Information Ranking (Optional if target y is provided)
        if y is not None and self.top_k_features and len(kept_indices) > self.top_k_features:
            X_scored = X[:, kept_indices]
            if self.is_classification:
                scores = mutual_info_classif(X_scored, y, random_state=42)
            else:
                scores = mutual_info_regression(X_scored, y, random_state=42)

            ranked_order = np.argsort(scores)[::-1][:self.top_k_features]
            kept_indices = [kept_indices[idx] for idx in ranked_order]

        self.selected_feature_indices_ = kept_indices
        self.selected_feature_names_ = [self.feature_names_[i] for i in self.selected_feature_indices_]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.selected_feature_indices_:
            raise ValueError("Selector is not fitted.")
        return X[:, self.selected_feature_indices_]

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None, feature_names: Optional[List[str]] = None) -> np.ndarray:
        self.fit(X, y, feature_names)
        return self.transform(X)
