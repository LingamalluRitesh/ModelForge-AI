"""
ModelForge AI - Model Monitoring & System Resource Schemas
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from datetime import datetime


class MonitoringMetricsQuery(BaseModel):
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    window_interval_minutes: int = 5


class ModelMonitoringMetricResponse(BaseModel):
    id: str
    deployment_id: str
    timestamp: datetime
    request_count: int
    error_count: int
    error_rate: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    observed_accuracy: Optional[float] = None
    observed_f1: Optional[float] = None
    observed_precision: Optional[float] = None
    observed_recall: Optional[float] = None
    prediction_distribution: Optional[Dict[str, Any]] = None
    confidence_distribution: Optional[Dict[str, Any]] = None

    model_config = {"from_attributes": True}


class SystemResourceResponse(BaseModel):
    timestamp: datetime
    cpu_percent: float
    memory_used_mb: float
    memory_total_mb: float
    memory_percent: float
    disk_used_gb: float
    disk_total_gb: float
    active_worker_count: int
    queued_task_count: int

    model_config = {"from_attributes": True}


class LiveDeploymentOverview(BaseModel):
    deployment_id: str
    name: str
    status: str
    is_healthy: bool
    current_rps: float
    avg_latency_ms: float
    error_rate: float
    last_24h_requests: int
