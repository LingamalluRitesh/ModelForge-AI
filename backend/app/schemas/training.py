"""
ModelForge AI - Training & AutoML Schemas
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class TrainingJobCreate(BaseModel):
    experiment_id: str
    name: str
    algorithm: str # logistic_regression, random_forest, xgboost, lightgbm, pytorch_mlp, etc.
    dataset_version_id: str
    target_column: str
    feature_columns: List[str]
    hyperparameters: Optional[Dict[str, Any]] = None
    validation_split: float = 0.2
    random_seed: int = 42


class TrainingJobResponse(BaseModel):
    id: str
    project_id: str
    experiment_id: str
    name: str
    algorithm: str
    dataset_version_id: str
    target_column: str
    feature_columns: List[str]
    hyperparameters: Optional[Dict[str, Any]] = None
    status: str
    progress_percentage: float
    current_epoch_or_step: int
    metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    run_id: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class AutoMLJobCreate(BaseModel):
    experiment_id: str
    name: str
    dataset_version_id: str
    target_column: str
    problem_type: Optional[str] = None # auto-detected if None
    optimization_metric: str = "f1"
    time_limit_seconds: int = 1800
    max_trials: int = 20
    algorithms_to_evaluate: Optional[List[str]] = None


class AutoMLTrialResponse(BaseModel):
    id: str
    trial_number: int
    algorithm_name: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, Any]
    primary_metric_value: float
    duration_seconds: float
    status: str
    run_id: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class AutoMLJobResponse(BaseModel):
    id: str
    project_id: str
    experiment_id: str
    name: str
    dataset_version_id: str
    target_column: str
    problem_type: str
    optimization_metric: str
    time_limit_seconds: int
    max_trials: int
    algorithms_to_evaluate: List[str]
    status: str
    best_algorithm: Optional[str] = None
    best_score: Optional[float] = None
    best_run_id: Optional[str] = None
    leaderboard: Optional[List[Dict[str, Any]]] = None
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    trials: List[AutoMLTrialResponse] = []

    model_config = {"from_attributes": True}
