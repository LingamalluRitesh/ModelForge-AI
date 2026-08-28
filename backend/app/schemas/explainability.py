"""
ModelForge AI - Explainable AI (SHAP) & Fairness Schemas
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class SHAPExplanationRequest(BaseModel):
    model_version_id: str
    sample_size: int = 100
    dataset_version_id: Optional[str] = None


class LocalExplanationRequest(BaseModel):
    model_version_id: str
    features: Dict[str, Any]


class LocalSHAPFactor(BaseModel):
    feature_name: str
    feature_value: Any
    shap_value: float
    percentage_contribution: float
    direction: str # positive, negative


class LocalExplanationResponse(BaseModel):
    model_version_id: str
    base_value: float
    prediction_value: float
    prediction_label: str
    factors: List[LocalSHAPFactor]


class SHAPExplanationResponse(BaseModel):
    id: str
    model_version_id: str
    explanation_type: str
    feature_importances: Dict[str, float]
    summary_plot_data: Optional[Dict[str, Any]] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class FairnessAnalysisRequest(BaseModel):
    model_version_id: str
    dataset_version_id: str
    sensitive_attribute: str # e.g. "gender", "age", "race"
    favorable_label: Any = 1


class FairnessReportResponse(BaseModel):
    id: str
    model_version_id: str
    sensitive_attribute: str
    demographic_parity_ratio: float
    disparate_impact_ratio: float
    equal_opportunity_difference: float
    equalized_odds_difference: float
    group_metrics: Dict[str, Any]
    is_fair: bool
    recommendations: List[str] = []
    created_at: datetime

    model_config = {"from_attributes": True}
