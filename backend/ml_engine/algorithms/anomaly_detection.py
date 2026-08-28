"""
ModelForge AI - Unsupervised Anomaly & Outlier Detection Algorithms
Implements Isolation Forest, Local Outlier Factor (LOF), One-Class SVM, and PyTorch Autoencoder.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


class IsolationForestAnomalyDetector:
    """
    Isolation Forest anomaly detector measuring tree path lengths to isolate points.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.05,
        random_state: int = 42,
    ):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray):
        self.model.fit(X)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns 1 for anomalies and 0 for inliers."""
        preds = self.model.predict(X)
        return np.where(preds == -1, 1, 0)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Returns anomaly severity score (higher = more anomalous)."""
        return -self.model.score_samples(X)


class LocalOutlierFactorDetector:
    """
    Local Outlier Factor (LOF) density-based anomaly detector.
    """

    def __init__(self, n_neighbors: int = 20, contamination: float = 0.05):
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray):
        self.model.fit(X)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.model.predict(X)
        return np.where(preds == -1, 1, 0)


class OneClassSVMDetector:
    """
    Support Vector Data Description / One-Class SVM.
    """

    def __init__(self, kernel: str = "rbf", nu: float = 0.05, gamma: str = "scale"):
        self.model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
        self.is_fitted = False

    def fit(self, X: np.ndarray):
        self.model.fit(X)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.model.predict(X)
        return np.where(preds == -1, 1, 0)
