"""
ModelForge AI - Feature Store Domain Service
Orchestrates entity definitions, feature views, online/offline sync schedules, and point-in-time training sets.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.exceptions import EntityNotFoundException, ValidationException
from app.models.feature import Feature
from app.models.project import Project
from app.schemas.feature import FeatureCreate, FeatureUpdate
from ml_engine.feature_store.feature_view import FeatureView, PointInTimeJoinEngine
from ml_engine.feature_store.entity_registry import OnlineFeatureStoreConnector, OfflineFeatureStoreConnector


class FeatureStoreService:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.online_connector = OnlineFeatureStoreConnector()
        self.offline_connector = OfflineFeatureStoreConnector()

    async def list_features(self, project_id: str, tag: Optional[str] = None) -> List[Feature]:
        query = select(Feature).where(Feature.project_id == project_id)
        res = await self.session.execute(query)
        features = list(res.scalars().all())
        if tag:
            features = [f for f in features if f.tags and tag in f.tags]
        return features

    async def create_feature(self, project_id: str, payload: FeatureCreate) -> Feature:
        feature = Feature(
            project_id=project_id,
            name=payload.name,
            data_type=payload.data_type,
            transformation_logic=payload.transformation_logic,
            description=payload.description,
            tags=payload.tags,
            validation_rules=payload.validation_rules,
        )
        self.session.add(feature)
        await self.session.commit()
        await self.session.refresh(feature)
        return feature

    async def sync_features_to_online_store(
        self,
        entity_name: str,
        records: List[Dict[str, Any]],
        entity_key: str,
    ) -> int:
        """Push a batch of updated feature values into Redis online store for real-time serving."""
        count = 0
        for rec in records:
            if entity_key in rec:
                eid = str(rec[entity_key])
                features_only = {k: v for k, v in rec.items() if k != entity_key}
                await self.online_connector.write_online_features(
                    entity_name=entity_name,
                    entity_id=eid,
                    feature_values=features_only,
                )
                count += 1
        return count

    async def get_online_feature_vector(
        self,
        entity_name: str,
        entity_ids: List[str],
        feature_names: List[str],
    ) -> List[Dict[str, Any]]:
        """Retrieve latest feature values from Redis online store."""
        return await self.online_connector.get_online_features(
            entity_name=entity_name,
            entity_ids=entity_ids,
            feature_names=feature_names,
        )
