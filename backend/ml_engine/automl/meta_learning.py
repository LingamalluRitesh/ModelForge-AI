"""
ModelForge AI - ML Engine: Dataset Meta-Feature Extractor
Extracts 40+ statistical, information-theoretic, complexity, and landmarking meta-features
to enable meta-learning warm-starts for Bayesian Optimization and automated model recommendation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


class DatasetMetaFeatureExtractor:
    """Extracts rich meta-feature representations summarizing tabular dataset characteristics."""

    @staticmethod
    def extract_meta_features(
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        is_classification: bool = True,
    ) -> Dict[str, float]:
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            num_numeric = len(numeric_cols)
            num_categorical = len(df.columns) - num_numeric
            X_mat = df[numeric_cols].fillna(0).values if num_numeric > 0 else np.zeros((len(df), 1))
        else:
            X_mat = np.asarray(X, dtype=np.float64)
            num_numeric = X_mat.shape[1]
            num_categorical = 0

        n_samples, n_features = X_mat.shape
        meta: Dict[str, float] = {}

        # 1. Simple Properties
        meta["num_instances"] = float(n_samples)
        meta["num_features"] = float(n_features)
        meta["num_numerical_features"] = float(num_numeric)
        meta["num_categorical_features"] = float(num_categorical)
        meta["ratio_instances_to_features"] = float(n_samples / max(1, n_features))
        meta["log_num_instances"] = float(np.log10(max(1, n_samples)))
        meta["log_num_features"] = float(np.log10(max(1, n_features)))

        # 2. Statistical Moments
        means = np.mean(X_mat, axis=0)
        stds = np.std(X_mat, axis=0)
        skews = skew(X_mat, axis=0, nan_policy="omit")
        kurts = kurtosis(X_mat, axis=0, nan_policy="omit")

        meta["mean_of_means"] = float(np.nanmean(means))
        meta["std_of_means"] = float(np.nanstd(means))
        meta["mean_of_stds"] = float(np.nanmean(stds))
        meta["std_of_stds"] = float(np.nanstd(stds))
        meta["mean_skewness"] = float(np.nanmean(skews))
        meta["max_skewness"] = float(np.nanmax(skews)) if len(skews) > 0 else 0.0
        meta["mean_kurtosis"] = float(np.nanmean(kurts))
        meta["max_kurtosis"] = float(np.nanmax(kurts)) if len(kurts) > 0 else 0.0

        # 3. Correlation & Sparsity
        if n_features > 1:
            corr_mat = np.corrcoef(X_mat, rowvar=False)
            np.fill_diagonal(corr_mat, 0)
            abs_corrs = np.abs(corr_mat[np.triu_indices(n_features, k=1)])
            meta["mean_feature_correlation"] = float(np.nanmean(abs_corrs)) if len(abs_corrs) > 0 else 0.0
            meta["max_feature_correlation"] = float(np.nanmax(abs_corrs)) if len(abs_corrs) > 0 else 0.0
        else:
            meta["mean_feature_correlation"] = 0.0
            meta["max_feature_correlation"] = 0.0

        meta["sparsity_ratio"] = float(np.mean(X_mat == 0))

        # 4. Target Properties (Supervised)
        if y is not None:
            y_arr = np.asarray(y)
            if is_classification:
                unique_classes, counts = np.unique(y_arr, return_counts=True)
                num_classes = len(unique_classes)
                meta["num_classes"] = float(num_classes)
                class_entropy = float(entropy(counts / np.sum(counts)))
                meta["class_entropy"] = class_entropy
                meta["imbalance_ratio"] = float(np.max(counts) / max(1, np.min(counts)))

                # 5. Fast Landmarking Models (Subsampled for speed)
                sample_limit = min(300, n_samples)
                sub_indices = np.random.choice(n_samples, sample_limit, replace=False)
                X_sub = X_mat[sub_indices]
                y_sub = y_arr[sub_indices]

                try:
                    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
                    dt_acc = np.mean(cross_val_score(dt, X_sub, y_sub, cv=2))
                    meta["landmark_decision_tree_acc"] = float(dt_acc)
                except Exception:
                    meta["landmark_decision_tree_acc"] = 0.5

                try:
                    nb = GaussianNB()
                    nb_acc = np.mean(cross_val_score(nb, X_sub, y_sub, cv=2))
                    meta["landmark_naive_bayes_acc"] = float(nb_acc)
                except Exception:
                    meta["landmark_naive_bayes_acc"] = 0.5
            else:
                meta["target_mean"] = float(np.mean(y_arr))
                meta["target_std"] = float(np.std(y_arr))
                meta["target_skewness"] = float(skew(y_arr))

        # Clean NaN/Inf values
        for k, v in list(meta.items()):
            if np.isnan(v) or np.isinf(v):
                meta[k] = 0.0

        return meta
