"""
ModelForge AI - ML Engine: Super Learner Stacking Ensemble Blender
Implements Van der Laan et al. Super Learner with K-Fold Out-of-Fold (OOF) cross-validation predictions
meta-feature matrix generation and non-negative least squares / Ridge meta-model blending.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class SuperLearnerStackingClassifier:
    """Multi-Model Out-Of-Fold Stacking Ensemble."""

    def __init__(self, base_models: List[Any], n_folds: int = 5):
        self.base_models = base_models
        self.n_folds = n_folds
        self.meta_weights_: Optional[np.ndarray] = None
        self.meta_intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        N, D = X.shape
        M = len(self.base_models)

        # Out-of-fold prediction matrix: (N, M)
        oof_predictions = np.zeros((N, M))

        indices = np.arange(N)
        folds = np.array_split(np.random.permutation(indices), self.n_folds)

        for fold_idx, test_idx in enumerate(folds):
            train_idx = np.setdiff1d(indices, test_idx)
            X_train, y_train = X[train_idx], y[train_idx]
            X_test = X[test_idx]

            for m_idx, model in enumerate(self.base_models):
                # Fit clone of model on train fold
                model.fit(X_train, y_train)
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_test)
                    pred = probs[:, 1] if probs.ndim == 2 else probs
                else:
                    pred = model.predict(X_test)
                oof_predictions[test_idx, m_idx] = pred

        # Fit all base models on complete training set
        for model in self.base_models:
            model.fit(X, y)

        # Meta-learner: Ridge regression on OOF matrix
        A = np.dot(oof_predictions.T, oof_predictions) + 1.0 * np.eye(M)
        b = np.dot(oof_predictions.T, y)
        self.meta_weights_ = np.linalg.solve(A, b)
        self.meta_intercept_ = float(np.mean(y) - np.dot(np.mean(oof_predictions, axis=0), self.meta_weights_))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Evaluate base models and combine predictions using meta-weights."""
        X = np.asarray(X, dtype=np.float64)
        N = X.shape[0]
        M = len(self.base_models)

        meta_features = np.zeros((N, M))
        for m_idx, model in enumerate(self.base_models):
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)
                meta_features[:, m_idx] = probs[:, 1] if probs.ndim == 2 else probs
            else:
                meta_features[:, m_idx] = model.predict(X)

        logits = np.dot(meta_features, self.meta_weights_) + self.meta_intercept_
        # Sigmoid
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -15.0, 15.0)))
        return np.column_stack([1.0 - probs, probs])

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)
