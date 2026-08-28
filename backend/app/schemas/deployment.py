"""
ModelForge AI - Deployment, Canary, A/B Testing & Routing Schemas
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class DeploymentCreate(BaseModel):
    name: str
    endpoint_path: str
    environment: str = "production"  # development, staging, production
    model_version_id: str
    strategy: str = "direct"         # direct, canary, ab_test, shadow
    secondary_model_version_id: Optional[str] = None
    primary_traffic_percentage: float = 100.0
    min_replicas: int = 1
    max_replicas: int = 5
    cpu_limit: str = "1000m"
    memory_limit: str = "2Gi"
    error_rate_threshold: float = 0.05
    latency_threshold_ms: float = 500.0
    auto_rollback_enabled: bool = True


class DeploymentUpdate(BaseModel):
    status: Optional[str] = None
    min_replicas: Optional[int] = None
    max_replicas: Optional[int] = None
    cpu_limit: Optional[str] = None
    memory_limit: Optional[str] = None
    error_rate_threshold: Optional[float] = None
    latency_threshold_ms: Optional[float] = None
    auto_rollback_enabled: Optional[bool] = None


class CanaryUpdateRequest(BaseModel):
    canary_stage_percentage: float = Field(..., ge=0.0, le=100.0) # e.g. 5, 20, 50, 100
    notes: Optional[str] = None


class ABTestTrafficUpdateRequest(BaseModel):
    primary_traffic_percentage: float = Field(..., ge=0.0, le=100.0)
    secondary_model_version_id: str


class RollbackRequest(BaseModel):
    target_model_version_id: str
    reason: str


class DeploymentRollbackResponse(BaseModel):
    id: str
    deployment_id: str
    from_model_version_id: str
    to_model_version_id: str
    trigger_type: str
    reason: str
    created_at: datetime

    model_config = {"from_attributes": True}


class DeploymentResponse(BaseModel):
    id: str
    project_id: str
    name: str
    endpoint_path: str
    environment: str
    status: str
    model_version_id: str
    strategy: str
    secondary_model_version_id: Optional[str] = None
    primary_traffic_percentage: float
    canary_stage_percentage: float
    min_replicas: int
    max_replicas: int
    current_replicas: int
    cpu_limit: str
    memory_limit: str
    is_healthy: bool
    error_rate_threshold: float
    latency_threshold_ms: float
    auto_rollback_enabled: bool
    created_at: datetime
    updated_at: datetime
    rollbacks: List[DeploymentRollbackResponse] = []

    model_config = {"from_attributes": True}
