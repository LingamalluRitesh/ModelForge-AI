"""
ModelForge AI - Advanced Ensemble Stacking & Blending
Implements multi-layer model stacking, out-of-fold feature generation, and meta-learner training.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge


class StackingClassifierModel:
    """
    Stacked Generalization Classifier combining diverse base estimators with meta-learner.
    """

    def __init__(
        self,
        base_models: List[Any],
        meta_model: Optional[Any] = None,
        n_splits: int = 5,
        use_probabilities: bool = True,
        random_state: int = 42,
    ):
        self.base_models = base_models
        self.meta_model = meta_model or LogisticRegression(C=1.0)
        self.n_splits = n_splits
        self.use_probabilities = use_probabilities
        self.random_state = random_state
        self.fitted_base_models_: List[Any] = []
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        n_samples = X.shape[0]
        n_models = len(self.base_models)

        # Generate Out-Of-Fold (OOF) predictions
        if self.use_probabilities:
            oof_features = np.zeros((n_samples, n_models))
        else:
            oof_features = np.zeros((n_samples, n_models))

        for model_idx, base_model in enumerate(self.base_models):
            for train_idx, val_idx in skf.split(X, y):
                X_train_fold, y_train_fold = X[train_idx], y[train_idx]
                X_val_fold = X[val_idx]

                import copy
                cloned = copy.deepcopy(base_model)
                cloned.fit(X_train_fold, y_train_fold)

                if self.use_probabilities and hasattr(cloned, "predict_proba"):
                    probs = cloned.predict_proba(X_val_fold)
                    oof_features[val_idx, model_idx] = probs[:, 1] if probs.shape[1] == 2 else np.argmax(probs, axis=1)
                else:
                    oof_features[val_idx, model_idx] = cloned.predict(X_val_fold)

        # Train meta-learner on OOF predictions
        self.meta_model.fit(oof_features, y)

        # Fit all base models on full dataset
        self.fitted_base_models_ = []
        for base_model in self.base_models:
            import copy
            cloned = copy.deepcopy(base_model)
            cloned.fit(X, y)
            self.fitted_base_models_.append(cloned)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        meta_features = self._construct_meta_features(X)
        return self.meta_model.predict(meta_features)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        meta_features = self._construct_meta_features(X)
        if hasattr(self.meta_model, "predict_proba"):
            return self.meta_model.predict_proba(meta_features)
        preds = self.predict(X)
        return np.column_stack([1 - preds, preds])

    def _construct_meta_features(self, X: np.ndarray) -> np.ndarray:
        meta_cols = []
        for model in self.fitted_base_models_:
            if self.use_probabilities and hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)
                meta_cols.append(probs[:, 1] if probs.shape[1] == 2 else np.argmax(probs, axis=1))
            else:
                meta_cols.append(model.predict(X))
        return np.column_stack(meta_cols)
