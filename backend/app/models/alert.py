"""
ModelForge AI - Alerting, Notifications & Immutable Audit Logging Models
"""

from datetime import datetime, timezone
import uuid
from typing import List, Optional
from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey, Integer, Float, String, Text, JSON
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from app.core.database import Base


class AlertSeverity(str):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(str):
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    MODEL_FAILURE = "model_failure"
    HIGH_LATENCY = "high_latency"
    HIGH_ERROR_RATE = "high_error_rate"
    LOW_ACCURACY = "low_accuracy"
    TRAINING_FAILURE = "training_failure"
    DEPLOYMENT_FAILURE = "deployment_failure"
    SECURITY_EVENT = "security_event"


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    
    alert_type: Mapped[str] = mapped_column(String(50), nullable=False)
    severity: Mapped[str] = mapped_column(String(20), default=AlertSeverity.WARNING, nullable=False)
    
    source_resource_type: Mapped[str] = mapped_column(String(50), nullable=False) # deployment, model, dataset, training_job
    source_resource_id: Mapped[str] = mapped_column(String(36), nullable=False)
    
    payload: Mapped[Optional[dict]] = mapped_column(JSON, default=dict, nullable=True)
    is_acknowledged: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_resolved: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    acknowledged_by_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("users.id"), nullable=True)
    acknowledged_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True, nullable=False
    )

    project = relationship("Project", back_populates="alerts")
    acknowledged_by = relationship("User")


class NotificationChannel(Base):
    __tablename__ = "notification_channels"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    channel_type: Mapped[str] = mapped_column(String(50), nullable=False) # webhook, email, slack, pagerduty
    
    config: Mapped[dict] = mapped_column(JSON, nullable=False) # webhook_url, email_recipients, token
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    subscribed_severities: Mapped[list] = mapped_column(JSON, default=list, nullable=False) # ["warning", "critical"]

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    project_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=True)
    
    action: Mapped[str] = mapped_column(String(100), index=True, nullable=False) # e.g., "project.create", "model.approve", "deployment.rollback"
    resource_type: Mapped[str] = mapped_column(String(50), index=True, nullable=False) # user, project, dataset, model, deployment
    resource_id: Mapped[Optional[str]] = mapped_column(String(36), index=True, nullable=True)
    
    request_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    
    status: Mapped[str] = mapped_column(String(20), default="SUCCESS", nullable=False) # SUCCESS, FAILURE, FORBIDDEN
    details: Mapped[Optional[dict]] = mapped_column(JSON, default=dict, nullable=True)

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True, nullable=False
    )

    user = relationship("User", back_populates="audit_logs")
    project = relationship("Project", back_populates="audit_logs")
