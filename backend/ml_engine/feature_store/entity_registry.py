"""
ModelForge AI - Feature Store: Entity Registry & Online / Offline Storage Connectors
Manages feature metadata, real-time KV online serving with Redis, and Parquet historical storage.
"""

import json
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime, timezone
import pandas as pd
from app.core.exceptions import ValidationException
from app.core.redis import get_redis_client


class EntityDefinition:
    """Represents a primary business domain entity (e.g., customer_id, merchant_id, device_id)."""

    def __init__(self, name: str, join_key: str, description: str = "", value_type: str = "string"):
        self.name = name
        self.join_key = join_key
        self.description = description
        self.value_type = value_type


class OnlineFeatureStoreConnector:
    """Ultra-low latency Redis KV storage connector for real-time online feature vectors."""

    def __init__(self, key_prefix: str = "mf_features"):
        self.key_prefix = key_prefix

    async def write_online_features(
        self,
        entity_name: str,
        entity_id: str,
        feature_values: Dict[str, Any],
        ttl_seconds: int = 86400 * 30, # 30 days default
    ):
        """Save latest entity feature map to Redis hash with TTL."""
        client = await get_redis_client()
        redis_key = f"{self.key_prefix}:{entity_name}:{entity_id}"
        
        # Serialize nested or complex types
        serialized = {
            k: (json.dumps(v) if isinstance(v, (dict, list)) else str(v))
            for k, v in feature_values.items()
        }
        await client.hset(redis_key, mapping=serialized)
        await client.expire(redis_key, ttl_seconds)

    async def get_online_features(
        self,
        entity_name: str,
        entity_ids: List[str],
        feature_names: List[str],
    ) -> List[Dict[str, Any]]:
        """Batch-fetch online feature vectors across entity IDs in sub-millisecond pipeline."""
        client = await get_redis_client()
        pipeline = client.pipeline()

        for eid in entity_ids:
            redis_key = f"{self.key_prefix}:{entity_name}:{eid}"
            pipeline.hmget(redis_key, feature_names)

        results = await pipeline.execute()
        
        feature_vectors = []
        for eid, row_values in zip(entity_ids, results):
            row_dict = {f_name: val for f_name, val in zip(feature_names, row_values)}
            row_dict["_entity_id"] = eid
            feature_vectors.append(row_dict)

        return feature_vectors


class OfflineFeatureStoreConnector:
    """Historical columnar Parquet connector for training set generation and offline batch scoring."""

    def __init__(self, storage_root: str = "data_storage/feature_store"):
        self.storage_root = storage_root

    def write_partitioned_parquet(
        self,
        feature_view_name: str,
        df: pd.DataFrame,
        partition_col: str = "year_month",
    ):
        import os
        from pathlib import Path
        view_path = Path(self.storage_root) / feature_view_name
        view_path.mkdir(parents=True, exist_ok=True)

        if partition_col in df.columns:
            df.to_parquet(view_path, partition_cols=[partition_col], index=False)
        else:
            df.to_parquet(view_path / "data.parquet", index=False)

    def read_feature_slice(
        self,
        feature_view_name: str,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        from pathlib import Path
        view_path = Path(self.storage_root) / feature_view_name
        if not view_path.exists():
            return pd.DataFrame()
        return pd.read_parquet(view_path, columns=columns)
