"""
ModelForge AI - Preprocessing: Categorical Encoders
Implements One-Hot, Ordinal, Frequency, and Target encoding with unseen category handling.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


class AdvancedCategoricalEncoder:
    """Categorical encoder supporting One-Hot, Ordinal, and Frequency encoding."""

    def __init__(self, method: str = "one_hot", max_categories: int = 50):
        self.method = method.lower()
        self.max_categories = max_categories
        self.encoder: Optional[Union[OneHotEncoder, OrdinalEncoder]] = None
        self.cat_cols: List[str] = []
        self.encoded_feature_names: List[str] = []
        self.frequency_maps: Dict[str, Dict[str, float]] = {}

    def fit(self, df: pd.DataFrame, target: Optional[pd.Series] = None) -> "AdvancedCategoricalEncoder":
        self.cat_cols = list(df.select_dtypes(include=["object", "category", "bool"]).columns)
        if not self.cat_cols:
            return self

        if self.method == "one_hot":
            self.encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                max_categories=self.max_categories,
            )
            self.encoder.fit(df[self.cat_cols].astype(str))
            self.encoded_feature_names = list(self.encoder.get_feature_names_out(self.cat_cols))

        elif self.method == "ordinal":
            self.encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
            self.encoder.fit(df[self.cat_cols].astype(str))
            self.encoded_feature_names = self.cat_cols

        elif self.method == "frequency":
            for col in self.cat_cols:
                freq = df[col].astype(str).value_counts(normalize=True).to_dict()
                self.frequency_maps[col] = freq
            self.encoded_feature_names = self.cat_cols

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.cat_cols:
            return df.copy()

        df_out = df.copy()
        if self.method == "one_hot" and self.encoder:
            encoded_arr = self.encoder.transform(df_out[self.cat_cols].astype(str))
            encoded_df = pd.DataFrame(encoded_arr, columns=self.encoded_feature_names, index=df_out.index)
            df_out = df_out.drop(columns=self.cat_cols)
            df_out = pd.concat([df_out, encoded_df], axis=1)

        elif self.method == "ordinal" and self.encoder:
            df_out[self.cat_cols] = self.encoder.transform(df_out[self.cat_cols].astype(str))

        elif self.method == "frequency":
            for col in self.cat_cols:
                freq_map = self.frequency_maps.get(col, {})
                df_out[col] = df_out[col].astype(str).map(lambda v: freq_map.get(v, 0.0))

        return df_out

    def fit_transform(self, df: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(df, target).transform(df)
