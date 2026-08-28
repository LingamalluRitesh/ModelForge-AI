"""
ModelForge AI - ML Engine: Advanced Multivariate Data Drift Detection
Implements Maximum Mean Discrepancy (MMD) with RBF kernels, Adversarial Validation (Domain Classifier),
and Energy Distance to detect joint multi-dimensional distribution shifts that marginal 1D tests miss.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.spatial.distance import cdist


class MultivariateDriftDetector:
    """Detects joint feature distribution drift using kernel two-sample tests and domain adversarial classifiers."""

    @staticmethod
    def compute_mmd(
        X_reference: np.ndarray,
        X_current: np.ndarray,
        gamma: Optional[float] = None,
        max_samples: int = 500,
    ) -> Dict[str, Any]:
        """
        Maximum Mean Discrepancy (MMD) with radial basis function (RBF) kernel:
        $MMD^2(P, Q) = \mathbb{E}[k(x, x')] + \mathbb{E}[k(y, y')] - 2\mathbb{E}[k(x, y)]$
        """
        X = np.asarray(X_reference, dtype=np.float64)
        Y = np.asarray(X_current, dtype=np.float64)

        # Downsample for memory and computational efficiency
        if len(X) > max_samples:
            X = X[np.random.choice(len(X), max_samples, replace=False)]
        if len(Y) > max_samples:
            Y = Y[np.random.choice(len(Y), max_samples, replace=False)]

        # Median heuristic for RBF bandwidth gamma
        if gamma is None:
            combined = np.vstack([X, Y])
            dists = cdist(combined, combined, metric="sqeuclidean")
            median_dist = np.median(dists)
            gamma = 1.0 / max(1e-4, 2.0 * median_dist)

        # Compute kernel matrices
        K_XX = np.exp(-gamma * cdist(X, X, metric="sqeuclidean"))
        K_YY = np.exp(-gamma * cdist(Y, Y, metric="sqeuclidean"))
        K_XY = np.exp(-gamma * cdist(X, Y, metric="sqeuclidean"))

        # Unbiased MMD squared estimator
        m = len(X)
        n = len(Y)
        np.fill_diagonal(K_XX, 0)
        np.fill_diagonal(K_YY, 0)

        mmd_sq = (
            np.sum(K_XX) / (m * (m - 1))
            + np.sum(K_YY) / (n * (n - 1))
            - 2.0 * np.sum(K_XY) / (m * n)
        )
        mmd_val = float(np.sqrt(max(0.0, mmd_sq)))

        # Threshold heuristic: MMD >= 0.05 implies noticeable distribution distance
        is_drifted = mmd_val >= 0.05

        return {
            "method": "Maximum Mean Discrepancy (MMD)",
            "mmd_distance": round(mmd_val, 4),
            "is_drifted": is_drifted,
            "kernel": "rbf",
            "gamma": round(float(gamma), 6),
            "reference_samples": m,
            "current_samples": n,
        }

    @staticmethod
    def compute_adversarial_validation_drift(
        X_reference: np.ndarray,
        X_current: np.ndarray,
        n_splits: int = 3,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Adversarial Validation: Trains a binary domain classifier to distinguish between
        Reference ($Y=0$) and Current ($Y=1$) data.
        If AUC $\approx 0.50$, distributions are identical. If AUC $> 0.70$, severe joint drift has occurred.
        """
        X_ref = np.asarray(X_reference, dtype=np.float64)
        X_cur = np.asarray(X_current, dtype=np.float64)

        X_all = np.vstack([X_ref, X_cur])
        y_all = np.concatenate([np.zeros(len(X_ref), dtype=int), np.ones(len(X_cur), dtype=int)])

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        oof_probs = np.zeros(len(y_all), dtype=np.float64)

        for train_idx, val_idx in skf.split(X_all, y_all):
            clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=random_state)
            clf.fit(X_all[train_idx], y_all[train_idx])
            oof_probs[val_idx] = clf.predict_proba(X_all[val_idx])[:, 1]

        auc = float(roc_auc_score(y_all, oof_probs))
        is_drifted = auc >= 0.68

        return {
            "method": "Adversarial Validation Domain Classifier",
            "domain_classifier_auc": round(auc, 4),
            "is_drifted": is_drifted,
            "drift_severity": "Critical" if auc >= 0.85 else ("Warning" if auc >= 0.68 else "None"),
            "interpretation": (
                "Distributions are indistinguishable (No Drift)" if auc < 0.68
                else "Domain classifier accurately distinguishes current traffic from baseline (Severe Drift)"
            ),
        }
