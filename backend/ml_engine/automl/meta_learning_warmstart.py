"""
ModelForge AI - AutoML: Meta-Learning Configuration Warmstarter
Extracts dataset meta-features (landmarkers, statistical moments, class entropy)
and predicts optimal initial hyperparameter configurations via nearest-dataset meta-learning.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class DatasetMetaFeatureExtractor:
    """Extracts meta-features from tabular datasets."""
    @staticmethod
    def extract_meta_features(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        N, D = X.shape

        return {
            "num_instances": float(N),
            "num_features": float(D),
            "feature_to_instance_ratio": float(D / max(1, N)),
            "mean_skewness": float(np.mean(np.abs(np.mean((X - np.mean(X, axis=0)) ** 3, axis=0)))),
            "mean_kurtosis": float(np.mean(np.mean((X - np.mean(X, axis=0)) ** 4, axis=0))),
            "target_entropy": float(-np.sum((np.unique(y, return_counts=True)[1] / N) * np.log2(np.clip(np.unique(y, return_counts=True)[1] / N, 1e-10, 1.0)))),
        }
