"""
ModelForge AI - Preprocessing: DateTime & Text Transformers
Extracts rich temporal components, cyclical sin/cos encodings, and TF-IDF / N-gram features.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class DateTimeFeatureExtractor:
    """Extracts year, month, day, dayofweek, hour, is_weekend, and cyclical sin/cos signals."""

    def __init__(self, date_columns: Optional[List[str]] = None, encode_cyclical: bool = True):
        self.date_columns = date_columns or []
        self.encode_cyclical = encode_cyclical

    def fit(self, df: pd.DataFrame) -> "DateTimeFeatureExtractor":
        if not self.date_columns:
            # Auto-detect datetime columns
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    self.date_columns.append(col)
                elif df[col].dtype == "object":
                    # Sample first 10 rows
                    sample = df[col].dropna().head(10)
                    try:
                        if len(sample) > 0:
                            pd.to_datetime(sample, errors="raise")
                            self.date_columns.append(col)
                    except Exception:
                        pass
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        for col in self.date_columns:
            if col not in df_out.columns:
                continue
            dt_series = pd.to_datetime(df_out[col], errors="coerce")

            df_out[f"{col}_year"] = dt_series.dt.year.fillna(2000).astype(int)
            df_out[f"{col}_month"] = dt_series.dt.month.fillna(1).astype(int)
            df_out[f"{col}_day"] = dt_series.dt.day.fillna(1).astype(int)
            df_out[f"{col}_dayofweek"] = dt_series.dt.dayofweek.fillna(0).astype(int)
            df_out[f"{col}_hour"] = dt_series.dt.hour.fillna(0).astype(int)
            df_out[f"{col}_is_weekend"] = dt_series.dt.dayofweek.isin([5, 6]).astype(int)

            if self.encode_cyclical:
                # Month cyclical (1-12)
                df_out[f"{col}_month_sin"] = np.sin(2 * np.pi * df_out[f"{col}_month"] / 12.0)
                df_out[f"{col}_month_cos"] = np.cos(2 * np.pi * df_out[f"{col}_month"] / 12.0)
                # Day of week cyclical (0-6)
                df_out[f"{col}_dow_sin"] = np.sin(2 * np.pi * df_out[f"{col}_dayofweek"] / 7.0)
                df_out[f"{col}_dow_cos"] = np.cos(2 * np.pi * df_out[f"{col}_dayofweek"] / 7.0)
                # Hour cyclical (0-23)
                df_out[f"{col}_hour_sin"] = np.sin(2 * np.pi * df_out[f"{col}_hour"] / 24.0)
                df_out[f"{col}_hour_cos"] = np.cos(2 * np.pi * df_out[f"{col}_hour"] / 24.0)

            df_out = df_out.drop(columns=[col])

        return df_out


class TextFeatureExtractor:
    """Transforms raw text columns using TF-IDF vectorization with n-gram extraction."""

    def __init__(self, text_columns: Optional[List[str]] = None, max_features: int = 50, ngram_range=(1, 2)):
        self.text_columns = text_columns or []
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizers: Dict[str, TfidfVectorizer] = {}

    def fit(self, df: pd.DataFrame) -> "TextFeatureExtractor":
        for col in self.text_columns:
            if col in df.columns:
                vec = TfidfVectorizer(
                    max_features=self.max_features,
                    ngram_range=self.ngram_range,
                    stop_words="english",
                )
                vec.fit(df[col].fillna("").astype(str))
                self.vectorizers[col] = vec
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        for col, vec in self.vectorizers.items():
            if col not in df_out.columns:
                continue
            sparse_mat = vec.transform(df_out[col].fillna("").astype(str))
            feature_names = [f"{col}_tfidf_{w}" for w in vec.get_feature_names_out()]
            tfidf_df = pd.DataFrame(sparse_mat.toarray(), columns=feature_names, index=df_out.index)
            df_out = df_out.drop(columns=[col])
            df_out = pd.concat([df_out, tfidf_df], axis=1)
        return df_out
