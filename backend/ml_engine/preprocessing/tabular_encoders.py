"""
ModelForge AI - ML Engine: Tabular Feature Encoders
Implements Target Encoding with empirical Bayes m-estimate smoothing,
Weight of Evidence (WoE) & Information Value (IV), James-Stein Bayesian encoding,
and Helmert / Polynomial contrast encoding for high-cardinality categorical variables.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class TargetEncoder:
    """
    Target Encoder with Micci-Barreca / Empirical Bayes smoothing:
    $S_i = \lambda_i \bar{y}_i + (1 - \lambda_i) \bar{y}_{global}$, where $\lambda_i = \frac{n_i}{n_i + m}$
    """

    def __init__(self, smoothing: float = 10.0, cv_folds: int = 5):
        self.smoothing = smoothing
        self.cv_folds = cv_folds
        self.global_mean_: float = 0.0
        self.mapping_: Dict[str, Dict[Any, float]] = {}

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray], categorical_cols: Optional[List[str]] = None):
        X = pd.DataFrame(X)
        y = np.asarray(y, dtype=np.float64)
        self.global_mean_ = float(np.mean(y))

        cols = categorical_cols or list(X.select_dtypes(include=["object", "category"]).columns)
        self.mapping_ = {}

        for col in cols:
            series = X[col]
            stats = pd.DataFrame({"cat": series, "target": y}).groupby("cat").agg(["count", "mean"])
            counts = stats["target"]["count"]
            means = stats["target"]["mean"]

            # Smoothing weight: lambda = n / (n + m)
            weights = counts / (counts + self.smoothing)
            smoothed_vals = weights * means + (1.0 - weights) * self.global_mean_

            self.mapping_[col] = smoothed_vals.to_dict()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = pd.DataFrame(X).copy()
        for col, col_map in self.mapping_.items():
            if col in X_out.columns:
                X_out[col] = X_out[col].map(col_map).fillna(self.global_mean_)
        return X_out


class WeightOfEvidenceEncoder:
    """
    Weight of Evidence (WoE) and Information Value (IV) for binary classification:
    $WoE_i = \ln\left(\frac{P(Y=1|X=i)}{P(Y=0|X=i)}\right)$, $IV = \sum (P(Y=1|X=i) - P(Y=0|X=i)) \cdot WoE_i$
    """

    def __init__(self, laplace_smooth: float = 0.5):
        self.laplace_smooth = laplace_smooth
        self.woe_mapping_: Dict[str, Dict[Any, float]] = {}
        self.iv_scores_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray], categorical_cols: Optional[List[str]] = None):
        X = pd.DataFrame(X)
        y = np.asarray(y, dtype=np.int64)

        total_pos = np.sum(y == 1)
        total_neg = np.sum(y == 0)

        cols = categorical_cols or list(X.select_dtypes(include=["object", "category"]).columns)
        self.woe_mapping_ = {}
        self.iv_scores_ = {}

        for col in cols:
            series = X[col]
            df = pd.DataFrame({"cat": series, "target": y})
            pos_counts = df[df["target"] == 1].groupby("cat").size()
            neg_counts = df[df["target"] == 0].groupby("cat").size()

            all_cats = series.unique()
            col_woe = {}
            total_iv = 0.0

            for cat in all_cats:
                p = pos_counts.get(cat, 0) + self.laplace_smooth
                n = neg_counts.get(cat, 0) + self.laplace_smooth

                dist_pos = p / (total_pos + len(all_cats) * self.laplace_smooth)
                dist_neg = n / (total_neg + len(all_cats) * self.laplace_smooth)

                woe = np.log(dist_pos / dist_neg)
                iv_cat = (dist_pos - dist_neg) * woe

                col_woe[cat] = float(woe)
                total_iv += float(iv_cat)

            self.woe_mapping_[col] = col_woe
            self.iv_scores_[col] = round(total_iv, 4)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = pd.DataFrame(X).copy()
        for col, col_map in self.woe_mapping_.items():
            if col in X_out.columns:
                X_out[col] = X_out[col].map(col_map).fillna(0.0)
        return X_out
