"""
ModelForge AI - Alerting, Notifications & Audit Log Schemas
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, EmailStr
from datetime import datetime


class AlertResponse(BaseModel):
    id: str
    project_id: str
    title: str
    message: str
    alert_type: str
    severity: str
    source_resource_type: str
    source_resource_id: str
    payload: Optional[Dict[str, Any]] = None
    is_acknowledged: bool
    is_resolved: bool
    acknowledged_by_id: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class AlertRuleCreate(BaseModel):
    name: str
    alert_type: str
    severity: str = "warning"
    condition_metric: str
    operator: str # >, <, >=, <=, ==
    threshold_value: float
    is_enabled: bool = True


class NotificationChannelCreate(BaseModel):
    name: str
    channel_type: str # webhook, email, slack
    config: Dict[str, Any]
    subscribed_severities: List[str] = ["warning", "critical"]
    is_enabled: bool = True


class NotificationChannelResponse(BaseModel):
    id: str
    project_id: str
    name: str
    channel_type: str
    config: Dict[str, Any]
    subscribed_severities: List[str]
    is_enabled: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class AuditLogResponse(BaseModel):
    id: str
    user_id: Optional[str] = None
    project_id: Optional[str] = None
    action: str
    resource_type: str
    resource_id: Optional[str] = None
    request_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    status: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime

    model_config = {"from_attributes": True}
