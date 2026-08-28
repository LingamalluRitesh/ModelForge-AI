"""
ModelForge AI - Deployment, Canary, A/B Testing & Routing Models
Supports real-time inference endpoints, Canary rollouts (5% -> 20% -> 50% -> 100%),
A/B traffic splitting, automated rollbacks, and multi-environment promotions.
"""

from datetime import datetime, timezone
import uuid
from typing import List, Optional
from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey, Integer, Float, String, Text, JSON
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from app.core.database import Base


class Deployment(Base):
    __tablename__ = "deployments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(150), index=True, nullable=False)
    endpoint_path: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False) # e.g. "fraud-detection-prod"
    
    environment: Mapped[str] = mapped_column(String(50), default="development", nullable=False) # development, staging, production
    status: Mapped[str] = mapped_column(String(50), default="active", nullable=False) # active, paused, scaling, updating, terminated
    
    # Active Primary Model
    model_version_id: Mapped[str] = mapped_column(String(36), ForeignKey("model_versions.id"), nullable=False)
    
    # Deployment Strategy
    strategy: Mapped[str] = mapped_column(String(50), default="direct", nullable=False) # direct, canary, ab_test, shadow
    
    # A/B Testing & Canary Parameters
    secondary_model_version_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("model_versions.id"), nullable=True)
    primary_traffic_percentage: Mapped[float] = mapped_column(Float, default=100.0, nullable=False)
    canary_stage_percentage: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    
    # Scaling & Resource Config
    min_replicas: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    max_replicas: Mapped[int] = mapped_column(Integer, default=5, nullable=False)
    current_replicas: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    cpu_limit: Mapped[str] = mapped_column(String(20), default="1000m", nullable=False)
    memory_limit: Mapped[str] = mapped_column(String(20), default="2Gi", nullable=False)
    
    # Health & Circuit Breaker
    is_healthy: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    error_rate_threshold: Mapped[float] = mapped_column(Float, default=0.05, nullable=False) # 5% max error rate
    latency_threshold_ms: Mapped[float] = mapped_column(Float, default=500.0, nullable=False)
    auto_rollback_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    
    created_by_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    project = relationship("Project", back_populates="deployments")
    model_version = relationship("ModelVersion", foreign_keys=[model_version_id], back_populates="deployments")
    secondary_model_version = relationship("ModelVersion", foreign_keys=[secondary_model_version_id])
    creator = relationship("User")
    predictions = relationship("PredictionLog", back_populates="deployment", cascade="all, delete-orphan")
    monitoring_metrics = relationship("ModelMonitoringMetric", back_populates="deployment", cascade="all, delete-orphan")
    drift_events = relationship("DriftEvent", back_populates="deployment", cascade="all, delete-orphan")
    rollbacks: Mapped[List["DeploymentRollbackHistory"]] = relationship("DeploymentRollbackHistory", back_populates="deployment", cascade="all, delete-orphan")


class DeploymentRollbackHistory(Base):
    __tablename__ = "deployment_rollback_history"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    deployment_id: Mapped[str] = mapped_column(String(36), ForeignKey("deployments.id", ondelete="CASCADE"), nullable=False)
    
    from_model_version_id: Mapped[str] = mapped_column(String(36), ForeignKey("model_versions.id"), nullable=False)
    to_model_version_id: Mapped[str] = mapped_column(String(36), ForeignKey("model_versions.id"), nullable=False)
    
    trigger_type: Mapped[str] = mapped_column(String(50), default="manual", nullable=False) # manual, automated_error_spike, automated_latency_spike, automated_drift
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    metrics_snapshot: Mapped[Optional[dict]] = mapped_column(JSON, default=dict, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    deployment: Mapped["Deployment"] = relationship("Deployment", back_populates="rollbacks")
    from_version = relationship("ModelVersion", foreign_keys=[from_model_version_id])
    to_version = relationship("ModelVersion", foreign_keys=[to_model_version_id])
