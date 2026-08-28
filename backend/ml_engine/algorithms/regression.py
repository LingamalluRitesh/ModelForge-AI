"""
ModelForge AI - ML Engine: Regression Algorithms
Implements Linear Regression, Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting, XGBoost, and LightGBM.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from app.core.exceptions import MLModelExecutionException


class BaseRegressor:
    """Base interface for all ModelForge regression models."""

    def __init__(self, **hyperparameters):
        self.hyperparameters = hyperparameters
        self.model = None
        self.feature_names_: Optional[List[str]] = None
        self.is_fitted: bool = False

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> "BaseRegressor":
        raise NotImplementedError

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self.is_fitted:
            raise MLModelExecutionException("Model is not fitted yet. Call fit() before predict().")
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)

    def get_feature_importances(self) -> Optional[Dict[str, float]]:
        if not self.is_fitted or not self.feature_names_:
            return None

        importances = None
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_)

        if importances is not None:
            total = np.sum(importances)
            norm_imp = importances / (total + 1e-12)
            return {
                name: float(imp)
                for name, imp in zip(self.feature_names_, norm_imp)
            }
        return None


class LinearRegressionModel(BaseRegressor):
    def __init__(self, fit_intercept: bool = True, **kwargs):
        super().__init__(fit_intercept=fit_intercept, **kwargs)
        self.model = LinearRegression(fit_intercept=fit_intercept)

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> "LinearRegressionModel":
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        self.model.fit(X, y)
        self.is_fitted = True
        return self


class RidgeRegressionModel(BaseRegressor):
    def __init__(self, alpha: float = 1.0, random_state: int = 42, **kwargs):
        super().__init__(alpha=alpha, random_state=random_state, **kwargs)
        self.model = Ridge(alpha=alpha, random_state=random_state)

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> "RidgeRegressionModel":
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        self.model.fit(X, y)
        self.is_fitted = True
        return self


class LassoRegressionModel(BaseRegressor):
    def __init__(self, alpha: float = 1.0, random_state: int = 42, **kwargs):
        super().__init__(alpha=alpha, random_state=random_state, **kwargs)
        self.model = Lasso(alpha=alpha, random_state=random_state)

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> "LassoRegressionModel":
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        self.model.fit(X, y)
        self.is_fitted = True
        return self


class RandomForestRegressorModel(BaseRegressor):
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 15,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        n_jobs: int = -1,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state,
            **kwargs
        )
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> "RandomForestRegressorModel":
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        self.model.fit(X, y)
        self.is_fitted = True
        return self


class GradientBoostingRegressorModel(BaseRegressor):
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
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            random_state=random_state,
        )

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> "GradientBoostingRegressorModel":
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        self.model.fit(X, y)
        self.is_fitted = True
        return self


class XGBoostRegressorModel(BaseRegressor):
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
            self.model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                random_state=random_state,
                **kwargs
            )
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state,
            )

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> "XGBoostRegressorModel":
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        self.model.fit(X, y)
        self.is_fitted = True
        return self


class LightGBMRegressorModel(BaseRegressor):
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
            self.model = lgb.LGBMRegressor(
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
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=random_state,
            )

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> "LightGBMRegressorModel":
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        self.model.fit(X, y)
        self.is_fitted = True
        return self


def get_regressor(algorithm_name: str, **hyperparameters) -> BaseRegressor:
    """Factory function for instantiating regressor by name."""
    algo_clean = algorithm_name.lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "linear_regression": LinearRegressionModel,
        "ridge": RidgeRegressionModel,
        "lasso": LassoRegressionModel,
        "random_forest_regression": RandomForestRegressorModel,
        "gradient_boosting_regression": GradientBoostingRegressorModel,
        "xgboost_regression": XGBoostRegressorModel,
        "lightgbm_regression": LightGBMRegressorModel,
    }
    if algo_clean in mapping:
        return mapping[algo_clean](**hyperparameters)
    raise ValueError(f"Unknown regressor algorithm '{algorithm_name}'. Available: {list(mapping.keys())}")
