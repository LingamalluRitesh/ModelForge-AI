"""
ModelForge AI - Model Registry, Governance & Approval Schemas
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class RegisteredModelCreate(BaseModel):
    name: str
    description: Optional[str] = None
    problem_type: str
    tags: Optional[List[str]] = None


class ModelVersionCreate(BaseModel):
    version_tag: str  # e.g. "v1.0.0"
    experiment_run_id: Optional[str] = None
    description: Optional[str] = None
    algorithm_name: Optional[str] = "xgboost"
    framework: Optional[str] = "xgboost"
    storage_uri: Optional[str] = "models/model.pkl"
    metrics: Optional[Dict[str, Any]] = None


class ModelApprovalRequestCreate(BaseModel):
    target_stage: str = "production"  # staging, production
    notes: Optional[str] = None


class ModelApprovalVote(BaseModel):
    action: str = "approve"  # approve, reject
    review_comments: Optional[str] = None


class ModelApprovalRequestResponse(BaseModel):
    id: str
    model_version_id: str
    target_stage: str
    status: str
    requested_by_id: str
    reviewed_by_id: Optional[str] = None
    review_comments: Optional[str] = None
    quality_checks_passed: bool
    created_at: datetime
    reviewed_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class ModelVersionResponse(BaseModel):
    id: str
    registered_model_id: str
    version_tag: str
    stage: str
    experiment_run_id: Optional[str] = None
    algorithm_name: str
    framework: str
    storage_uri: str
    signature: Optional[Dict[str, Any]] = None
    metrics: Dict[str, Any]
    hyperparameters: Optional[Dict[str, Any]] = None
    quality_gate_passed: bool
    quality_gate_summary: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class RegisteredModelResponse(BaseModel):
    id: str
    project_id: str
    name: str
    description: Optional[str] = None
    problem_type: str
    tags: Optional[List[str]] = None
    is_archived: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class QualityGateConfigSchema(BaseModel):
    name: str
    target_stage: str = "production"
    rules: List[Dict[str, Any]]  # e.g. [{"metric": "f1", "operator": ">=", "threshold": 0.85}]
    is_blocking: bool = True
