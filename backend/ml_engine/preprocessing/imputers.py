"""
ModelForge AI - Preprocessing: Imputers & Scalers
Handles missing value imputation (mean, median, most_frequent, constant, KNN)
and numeric scaling (standard, minmax, robust, maxabs).
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler


class AdvancedImputer:
    """Configurable multi-strategy imputer for numeric and categorical columns."""

    def __init__(self, numeric_strategy: str = "median", categorical_strategy: str = "most_frequent", fill_value: Any = None):
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.fill_value = fill_value
        self.num_imputer: Optional[SimpleImputer] = None
        self.cat_imputer: Optional[SimpleImputer] = None
        self.num_cols: List[str] = []
        self.cat_cols: List[str] = []

    def fit(self, df: pd.DataFrame) -> "AdvancedImputer":
        self.num_cols = list(df.select_dtypes(include=[np.number]).columns)
        self.cat_cols = [c for c in df.columns if c not in self.num_cols]

        if self.num_cols:
            self.num_imputer = SimpleImputer(strategy=self.numeric_strategy, fill_value=self.fill_value)
            self.num_imputer.fit(df[self.num_cols])

        if self.cat_cols:
            self.cat_imputer = SimpleImputer(strategy=self.categorical_strategy, fill_value=self.fill_value or "missing")
            self.cat_imputer.fit(df[self.cat_cols].astype(str))

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        if self.num_cols and self.num_imputer:
            df_out[self.num_cols] = self.num_imputer.transform(df_out[self.num_cols])
        if self.cat_cols and self.cat_imputer:
            df_out[self.cat_cols] = self.cat_imputer.transform(df_out[self.cat_cols].astype(str))
        return df_out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


class AdvancedScaler:
    """Configurable feature scaler supporting standard, minmax, robust, and maxabs."""

    def __init__(self, method: str = "standard"):
        self.method = method.lower()
        if self.method == "standard":
            self.scaler = StandardScaler()
        elif self.method == "minmax":
            self.scaler = MinMaxScaler()
        elif self.method == "robust":
            self.scaler = RobustScaler()
        elif self.method == "maxabs":
            self.scaler = MaxAbsScaler()
        else:
            raise ValueError(f"Unknown scaling method '{method}'")
        self.num_cols: List[str] = []

    def fit(self, df: pd.DataFrame) -> "AdvancedScaler":
        self.num_cols = list(df.select_dtypes(include=[np.number]).columns)
        if self.num_cols:
            self.scaler.fit(df[self.num_cols])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        if self.num_cols:
            df_out[self.num_cols] = self.scaler.transform(df_out[self.num_cols])
        return df_out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
