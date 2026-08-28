"""
ModelForge AI - ML Engine: Classification Algorithms
Implements production-grade classifiers using Scikit-Learn, XGBoost, LightGBM, and PyTorch.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from app.core.exceptions import MLModelExecutionException


class BaseClassifier:
    """Base interface for all ModelForge classification models."""

    def __init__(self, **hyperparameters):
        self.hyperparameters = hyperparameters
        self.model = None
        self.classes_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None
        self.is_fitted: bool = False

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> "BaseClassifier":
        raise NotImplementedError

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self.is_fitted:
            raise MLModelExecutionException("Model is not fitted yet. Call fit() before predict().")
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self.is_fitted:
            raise MLModelExecutionException("Model is not fitted yet. Call fit() before predict_proba().")
        if isinstance(X, pd.DataFrame):
            X = X.values
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        elif hasattr(self.model, "decision_function"):
            df = self.model.decision_function(X)
            # Sigmoid or softmax approximation
            if df.ndim == 1:
                prob = 1.0 / (1.0 + np.exp(-df))
                return np.vstack([1.0 - prob, prob]).T
            else:
                exp_df = np.exp(df - np.max(df, axis=1, keepdims=True))
                return exp_df / np.sum(exp_df, axis=1, keepdims=True)
        else:
            preds = self.predict(X)
            one_hot = np.zeros((len(preds), len(self.classes_)))
            for i, p in enumerate(preds):
                idx = np.where(self.classes_ == p)[0][0]
                one_hot[i, idx] = 1.0
            return one_hot

    def get_feature_importances(self) -> Optional[Dict[str, float]]:
        """Extract feature importances if available on fitted estimator."""
        if not self.is_fitted or not self.feature_names_:
            return None

        importances = None
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.mean(np.abs(self.model.coef_), axis=0)

        if importances is not None:
            # Normalize to sum = 1.0
            total = np.sum(importances)
            norm_imp = importances / (total + 1e-12)
            return {
                name: float(imp)
                for name, imp in zip(self.feature_names_, norm_imp)
            }
        return None


class LogisticRegressionClassifier(BaseClassifier):
    """Production Logistic Regression with L1/L2 regularization and elasticnet support."""

    def __init__(
        self,
        C: float = 1.0,
        penalty: str = "l2",
        solver: str = "lbfgs",
        max_iter: int = 1000,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(C=C, penalty=penalty, solver=solver, max_iter=max_iter, random_state=random_state, **kwargs)
        self.model = LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            max_iter=max_iter,
            random_state=random_state,
        )

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> "LogisticRegressionClassifier":
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        self.is_fitted = True
        return self


class DecisionTreeModel(BaseClassifier):
    """Decision Tree Classifier with max_depth, min_samples_split, and criterion control."""

    def __init__(
        self,
        max_depth: Optional[int] = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = "gini",
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=random_state,
            **kwargs
        )
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=random_state,
        )

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> "DecisionTreeModel":
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        self.is_fitted = True
        return self


class RandomForestModel(BaseClassifier):
    """Random Forest Classifier with parallel tree ensembles and out-of-bag scoring."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 15,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = "gini",
        n_jobs: int = -1,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            n_jobs=n_jobs,
            random_state=random_state,
            **kwargs
        )
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> "RandomForestModel":
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        self.is_fitted = True
        return self


class GradientBoostingModel(BaseClassifier):
    """Scikit-learn Gradient Boosting Classifier."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        subsample: float = 1.0,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            random_state=random_state,
            **kwargs
        )
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            random_state=random_state,
        )

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> "GradientBoostingModel":
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        self.is_fitted = True
        return self


class XGBoostClassifierModel(BaseClassifier):
    """Enterprise XGBoost Classifier."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            **kwargs
        )
        if xgb is not None:
            self.model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                random_state=random_state,
                eval_metric="logloss",
                **kwargs
            )
        else:
            # Fallback to GradientBoosting if xgboost binary not installed
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state,
            )

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> "XGBoostClassifierModel":
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        self.is_fitted = True
        return self


class LightGBMClassifierModel(BaseClassifier):
    """Enterprise LightGBM Classifier."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        max_depth: int = -1,
        subsample: float = 0.8,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            subsample=subsample,
            random_state=random_state,
            **kwargs
        )
        if lgb is not None:
            self.model = lgb.LGBMClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                max_depth=max_depth,
                subsample=subsample,
                random_state=random_state,
                verbosity=-1,
                **kwargs
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
            )

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> "LightGBMClassifierModel":
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        self.is_fitted = True
        return self


def get_classifier(algorithm_name: str, **hyperparameters) -> BaseClassifier:
    """Factory function for instantiating classifier by name."""
    algo_clean = algorithm_name.lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "logistic_regression": LogisticRegressionClassifier,
        "decision_tree": DecisionTreeModel,
        "random_forest": RandomForestModel,
        "gradient_boosting": GradientBoostingModel,
        "xgboost": XGBoostClassifierModel,
        "lightgbm": LightGBMClassifierModel,
    }
    if algo_clean in mapping:
        return mapping[algo_clean](**hyperparameters)
    raise ValueError(f"Unknown classifier algorithm '{algorithm_name}'. Available: {list(mapping.keys())}")
