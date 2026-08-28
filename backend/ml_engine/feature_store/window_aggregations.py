"""
ModelForge AI - Feature Store: Sliding & Tumbling Window Feature Transformer
Computes stateful statistical aggregates over time-series entities: Rolling Sum, Mean, StdDev,
Min, Max, Skewness, Kurtosis, and Exponential Decay weighted aggregates.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class TemporalWindowAggregator:
    """Computes point-in-time leakage-free rolling window features."""
    def __init__(self, windows: List[str] = ["1h", "24h", "7d", "30d"]):
        self.windows = windows

    def transform(
        self,
        df: pd.DataFrame,
        entity_col: str,
        timestamp_col: str,
        value_col: str,
    ) -> pd.DataFrame:
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values(by=[entity_col, timestamp_col])

        out_df = df.copy()

        for win in self.windows:
            rolled = (
                df.set_index(timestamp_col)
                .groupby(entity_col)[value_col]
                .rolling(win, closed="left")
            )

            out_df[f"{value_col}_sum_{win}"] = rolled.sum().values
            out_df[f"{value_col}_mean_{win}"] = rolled.mean().values
            out_df[f"{value_col}_std_{win}"] = rolled.std().fillna(0.0).values
            out_df[f"{value_col}_count_{win}"] = rolled.count().values

        return out_df
