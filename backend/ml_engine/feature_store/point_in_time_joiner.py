"""
ModelForge AI - Feature Store: Strict Point-in-Time (AS-OF) Feature Matrix Joiner
Constructs ML training datasets without future lookahead data leakage by joining historical entity records.
"""

from typing import Any, Dict, List, Optional
import pandas as pd


class PointInTimeJoiner:
    """Executes time-travel leakage-free join between observation labels and feature tables."""
    @staticmethod
    def as_of_join(
        observations_df: pd.DataFrame,
        features_df: pd.DataFrame,
        entity_id_col: str,
        timestamp_col: str,
    ) -> pd.DataFrame:
        obs = observations_df.copy()
        feat = features_df.copy()

        obs[timestamp_col] = pd.to_datetime(obs[timestamp_col])
        feat[timestamp_col] = pd.to_datetime(feat[timestamp_col])

        obs = obs.sort_values(by=timestamp_col)
        feat = feat.sort_values(by=timestamp_col)

        # Merge AS-OF: observation_time >= feature_time
        merged = pd.merge_asof(
            obs,
            feat,
            on=timestamp_col,
            by=entity_id_col,
            direction="backward",
        )
        return merged
