"""
ModelForge AI - Feature Store: Point-In-Time (AS-OF) Feature Join Engine
Guarantees zero future-information leakage for historical training dataset generation
by performing exact AS-OF temporal joins between entity observation timestamps and feature changelogs.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from app.core.exceptions import ValidationException


class FeatureView:
    """Represents a logically grouped schema of versioned features tied to an Entity."""

    def __init__(
        self,
        name: str,
        entity_id_column: str,
        timestamp_column: str,
        feature_columns: List[str],
        ttl_seconds: Optional[int] = None,
    ):
        self.name = name
        self.entity_id_column = entity_id_column
        self.timestamp_column = timestamp_column
        self.feature_columns = feature_columns
        self.ttl_seconds = ttl_seconds


class PointInTimeJoinEngine:
    """High-performance AS-OF temporal join engine preventing lookahead data leakage."""

    @staticmethod
    def as_of_join(
        entity_df: pd.DataFrame,
        feature_view_df: pd.DataFrame,
        entity_key: str,
        entity_timestamp_col: str,
        feature_timestamp_col: str,
        feature_columns: List[str],
        ttl_seconds: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Merge feature records into observation entities such that:
        $feature\_timestamp \le entity\_timestamp$ and $entity\_timestamp - feature\_timestamp \le TTL$.
        """
        # Ensure timestamp columns are parsed to UTC datetime
        e_df = entity_df.copy()
        f_df = feature_view_df.copy()

        e_df[entity_timestamp_col] = pd.to_datetime(e_df[entity_timestamp_col], utc=True)
        f_df[feature_timestamp_col] = pd.to_datetime(f_df[feature_timestamp_col], utc=True)

        # Sort both datasets by timestamp required by merge_asof
        e_df = e_df.sort_values(entity_timestamp_col).reset_index(drop=True)
        f_df = f_df.sort_values(feature_timestamp_col).reset_index(drop=True)

        # Subset feature columns
        f_subset = f_df[[entity_key, feature_timestamp_col] + feature_columns]

        tolerance = pd.Timedelta(seconds=ttl_seconds) if ttl_seconds is not None else None

        merged = pd.merge_asof(
            e_df,
            f_subset,
            left_on=entity_timestamp_col,
            right_on=feature_timestamp_col,
            by=entity_key,
            direction="backward",
            tolerance=tolerance,
        )

        return merged

    @staticmethod
    def get_historical_features(
        entity_df: pd.DataFrame,
        entity_key: str,
        entity_timestamp_col: str,
        feature_views: List[Tuple[FeatureView, pd.DataFrame]],
    ) -> pd.DataFrame:
        """Join multiple feature views onto entity observation dataframe with point-in-time correctness."""
        result_df = entity_df.copy()

        for fv, fv_df in feature_views:
            result_df = PointInTimeJoinEngine.as_of_join(
                entity_df=result_df,
                feature_view_df=fv_df,
                entity_key=entity_key,
                entity_timestamp_col=entity_timestamp_col,
                feature_timestamp_col=fv.timestamp_column,
                feature_columns=fv.feature_columns,
                ttl_seconds=fv.ttl_seconds,
            )

        return result_df
