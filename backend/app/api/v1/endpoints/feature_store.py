"""
ModelForge AI - Feature Store & Online KV Serving Endpoints
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db, get_current_user
from app.models.user import User
from app.schemas.common import APIResponse
from app.schemas.feature import FeatureCreate, FeatureResponse
from app.services.feature_store_service import FeatureStoreService

router = APIRouter(prefix="/feature-store", tags=["Feature Store & Entity Registry"])


class OnlineFeatureSyncRequest(BaseModel):
    entity_name: str
    entity_key: str
    records: List[Dict[str, Any]]


class OnlineFeatureFetchRequest(BaseModel):
    entity_name: str
    entity_ids: List[str]
    feature_names: List[str]


@router.get("/features", response_model=APIResponse[List[FeatureResponse]])
async def list_features(
    project_id: str = Query(...),
    tag: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    svc = FeatureStoreService(db)
    features = await svc.list_features(project_id, tag=tag)
    return APIResponse(data=features)


@router.post("/features", response_model=APIResponse[FeatureResponse])
async def create_feature(
    payload: FeatureCreate,
    project_id: str = Query(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    svc = FeatureStoreService(db)
    feat = await svc.create_feature(project_id, payload)
    return APIResponse(data=feat, message="Feature created successfully.")


@router.post("/online-sync", response_model=APIResponse[Dict[str, Any]])
async def sync_online_features(
    payload: OnlineFeatureSyncRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    svc = FeatureStoreService(db)
    count = await svc.sync_features_to_online_store(
        entity_name=payload.entity_name,
        records=payload.records,
        entity_key=payload.entity_key,
    )
    return APIResponse(data={"synced_records_count": count}, message="Online features synchronized to Redis store.")


@router.post("/online-fetch", response_model=APIResponse[List[Dict[str, Any]]])
async def fetch_online_features(
    payload: OnlineFeatureFetchRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    svc = FeatureStoreService(db)
    vectors = await svc.get_online_feature_vector(
        entity_name=payload.entity_name,
        entity_ids=payload.entity_ids,
        feature_names=payload.feature_names,
    )
    return APIResponse(data=vectors)
