"""
ModelForge AI - Feature Store Schemas
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class FeatureCreate(BaseModel):
    name: str
    entity_name: str = "default"
    description: Optional[str] = None
    data_type: str  # float, int, string, boolean, timestamp, vector
    transformation_type: str = "identity" # standard_scaler, one_hot, log, tfidf, custom
    transformation_params: Optional[Dict[str, Any]] = None
    source_dataset_id: Optional[str] = None
    source_column: Optional[str] = None
    tags: Optional[List[str]] = None


class FeatureUpdate(BaseModel):
    description: Optional[str] = None
    status: Optional[str] = None
    tags: Optional[List[str]] = None


class FeatureVersionResponse(BaseModel):
    id: str
    feature_id: str
    version_tag: str
    statistics: Optional[Dict[str, Any]] = None
    lineage_metadata: Optional[Dict[str, Any]] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class FeatureResponse(BaseModel):
    id: str
    project_id: str
    name: str
    entity_name: str
    description: Optional[str] = None
    data_type: str
    transformation_type: str
    transformation_params: Optional[Dict[str, Any]] = None
    source_dataset_id: Optional[str] = None
    source_column: Optional[str] = None
    status: str
    tags: Optional[List[str]] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class FeatureViewCreate(BaseModel):
    name: str
    description: Optional[str] = None
    entity_id: str
    feature_names: List[str]
    ttl_seconds: int = 86400


class FeatureViewResponse(BaseModel):
    id: str
    project_id: str
    name: str
    description: Optional[str] = None
    entity_id: str
    feature_names: List[str]
    ttl_seconds: int
    created_at: datetime

    model_config = {"from_attributes": True}
