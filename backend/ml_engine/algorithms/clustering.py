"""
ModelForge AI - ML Engine: Clustering Algorithms
Implements K-Means, DBSCAN, and Agglomerative Hierarchical Clustering with silhouette scoring.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from app.core.exceptions import MLModelExecutionException


class BaseClusteringModel:
    def __init__(self, **hyperparameters):
        self.hyperparameters = hyperparameters
        self.model = None
        self.labels_: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit_predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        raise NotImplementedError

    def compute_clustering_metrics(self, X: Union[np.ndarray, pd.DataFrame]) -> Dict[str, float]:
        if not self.is_fitted or self.labels_ is None:
            raise MLModelExecutionException("Clustering model not fitted.")
        if isinstance(X, pd.DataFrame):
            X = X.values

        unique_labels = set(self.labels_)
        # Filter out noise label (-1) for DBSCAN if present
        valid_labels = [l for l in unique_labels if l != -1]

        if len(valid_labels) < 2:
            return {
                "n_clusters": len(valid_labels),
                "silhouette_score": 0.0,
                "davies_bouldin_index": 0.0,
                "calinski_harabasz_score": 0.0,
            }

        # Filter noise points for silhouette calculation
        mask = self.labels_ != -1 if -1 in unique_labels else np.ones(len(self.labels_), dtype=bool)
        X_valid = X[mask]
        labels_valid = self.labels_[mask]

        sil = float(silhouette_score(X_valid, labels_valid))
        db = float(davies_bouldin_score(X_valid, labels_valid))
        ch = float(calinski_harabasz_score(X_valid, labels_valid))

        return {
            "n_clusters": len(valid_labels),
            "silhouette_score": sil,
            "davies_bouldin_index": db,
            "calinski_harabasz_score": ch,
        }


class KMeansClusteringModel(BaseClusteringModel):
    def __init__(self, n_clusters: int = 3, max_iter: int = 300, random_state: int = 42, **kwargs):
        super().__init__(n_clusters=n_clusters, max_iter=max_iter, random_state=random_state, **kwargs)
        self.model = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=random_state, n_init="auto")

    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> "KMeansClusteringModel":
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.model.fit(X)
        self.labels_ = self.model.labels_
        self.is_fitted = True
        return self

    def fit_predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        self.fit(X)
        return self.labels_

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self.is_fitted:
            raise MLModelExecutionException("Model not fitted.")
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)


class DBSCANClusteringModel(BaseClusteringModel):
    def __init__(self, eps: float = 0.5, min_samples: int = 5, **kwargs):
        super().__init__(eps=eps, min_samples=min_samples, **kwargs)
        self.model = DBSCAN(eps=eps, min_samples=min_samples)

    def fit_predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.labels_ = self.model.fit_predict(X)
        self.is_fitted = True
        return self.labels_


class AgglomerativeClusteringModel(BaseClusteringModel):
    def __init__(self, n_clusters: int = 3, linkage: str = "ward", **kwargs):
        super().__init__(n_clusters=n_clusters, linkage=linkage, **kwargs)
        self.model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)

    def fit_predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.labels_ = self.model.fit_predict(X)
        self.is_fitted = True
        return self.labels_
