"""
ModelForge AI - Data Drift & Concept Drift Schemas
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class DriftAnalysisRequest(BaseModel):
    deployment_id: str
    target_dataset_version_id: Optional[str] = None # Or uses recent live inference logs
    baseline_dataset_version_id: Optional[str] = None # Defaults to training dataset
    psi_threshold_warning: float = 0.1
    psi_threshold_critical: float = 0.25
    sample_size: int = 1000


class FeatureDriftDetail(BaseModel):
    feature_name: str
    drift_score: float # PSI or normalized distance
    algorithm_used: str # psi, ks_test, chi_square, wasserstein
    p_value: Optional[float] = None
    has_drifted: bool
    severity: str # none, low, medium, high, critical
    baseline_distribution: Optional[Dict[str, Any]] = None
    target_distribution: Optional[Dict[str, Any]] = None


class DriftReportResponse(BaseModel):
    id: str
    deployment_id: str
    drift_type: str
    severity: str
    overall_drift_score: float
    drifted_features_count: int
    total_features_count: int
    baseline_dataset_version_id: Optional[str] = None
    target_sample_size: int
    feature_metrics: Dict[str, FeatureDriftDetail]
    recommendations: List[str] = []
    is_resolved: bool
    retraining_triggered: bool
    created_at: datetime

    model_config = {"from_attributes": True}
