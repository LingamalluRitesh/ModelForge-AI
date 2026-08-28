"""
ModelForge AI - Real-Time Inference & Batch Prediction Schemas
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime


class RealtimePredictionRequest(BaseModel):
    features: Dict[str, Any]
    model_version_tag: Optional[str] = None # Optional override, otherwise router selects


class BatchPredictionRequest(BaseModel):
    instances: List[Dict[str, Any]]
    model_version_tag: Optional[str] = None


class RealtimePredictionResponse(BaseModel):
    prediction: Union[int, float, str, List[Any], Dict[str, Any]]
    probability_or_confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    model_name: str
    model_version_tag: str
    deployment_id: str
    request_id: str
    latency_ms: float
    timestamp: datetime


class GroundTruthFeedbackRequest(BaseModel):
    request_id: str
    ground_truth: Any


class BatchInferenceJobCreate(BaseModel):
    name: str
    model_version_id: str
    dataset_version_id: str


class BatchInferenceJobResponse(BaseModel):
    id: str
    project_id: str
    model_version_id: str
    dataset_version_id: str
    name: str
    status: str
    total_rows: int
    processed_rows: int
    failed_rows: int
    output_storage_uri: Optional[str] = None
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

    model_config = {"from_attributes": True}
