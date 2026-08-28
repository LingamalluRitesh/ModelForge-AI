"""
ModelForge AI - Experiment Tracking & Run Schemas
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class ExperimentCreate(BaseModel):
    name: str
    description: Optional[str] = None
    tags: Optional[List[str]] = None


class ExperimentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    is_archived: Optional[bool] = None


class ExperimentRunCreate(BaseModel):
    name: str
    dataset_version_id: Optional[str] = None
    algorithm_name: str
    framework: str = "scikit-learn"
    hyperparameters: Optional[Dict[str, Any]] = None
    code_version_hash: Optional[str] = None


class ExperimentRunResponse(BaseModel):
    id: str
    experiment_id: str
    name: str
    dataset_version_id: Optional[str] = None
    algorithm_name: str
    framework: str
    status: str
    duration_seconds: Optional[float] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    system_metrics: Optional[Dict[str, Any]] = None
    model_artifact_uri: Optional[str] = None
    code_version_hash: Optional[str] = None
    error_traceback: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class ExperimentResponse(BaseModel):
    id: str
    project_id: str
    name: str
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    is_archived: bool
    created_at: datetime
    updated_at: datetime
    runs_count: Optional[int] = 0

    model_config = {"from_attributes": True}


class RunComparisonRequest(BaseModel):
    run_ids: List[str] = Field(..., min_length=2, max_length=10)


class RunComparisonResponse(BaseModel):
    runs: List[ExperimentRunResponse]
    metric_keys: List[str]
    hyperparameter_keys: List[str]
    best_run_id_per_metric: Dict[str, str]
