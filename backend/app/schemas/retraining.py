"""
ModelForge AI - Automated Retraining Schemas
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class RetrainingPolicyCreate(BaseModel):
    name: str
    target_registered_model_id: str
    trigger_type: str = "drift_threshold" # drift_threshold, accuracy_threshold, schedule, new_dataset
    drift_score_threshold: float = 0.25
    performance_drop_threshold: float = 0.05
    cron_schedule: Optional[str] = None
    auto_promote_if_passed: bool = False
    min_improvement_margin: float = 0.01


class RetrainingTriggerRequest(BaseModel):
    policy_id: str
    reason: str = "Manual retraining execution trigger"


class RetrainingExecutionResponse(BaseModel):
    id: str
    policy_id: str
    drift_event_id: Optional[str] = None
    status: str
    trigger_reason: str
    baseline_model_version_id: str
    new_candidate_model_version_id: Optional[str] = None
    baseline_metric_score: float
    challenger_metric_score: Optional[float] = None
    comparison_summary: Optional[Dict[str, Any]] = None
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class RetrainingPolicyResponse(BaseModel):
    id: str
    project_id: str
    name: str
    is_active: bool
    trigger_type: str
    drift_score_threshold: float
    performance_drop_threshold: float
    cron_schedule: Optional[str] = None
    target_registered_model_id: str
    auto_promote_if_passed: bool
    min_improvement_margin: float
    created_at: datetime
    executions: List[RetrainingExecutionResponse] = []

    model_config = {"from_attributes": True}
