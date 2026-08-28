"""
ModelForge AI - Automated Retraining Policies & Trigger Execution Models
"""

from datetime import datetime, timezone
import uuid
from typing import List, Optional
from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey, Integer, Float, String, Text, JSON
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from app.core.database import Base


class RetrainingPolicy(Base):
    __tablename__ = "retraining_policies"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    trigger_type: Mapped[str] = mapped_column(String(50), default="drift_threshold", nullable=False) # drift_threshold, accuracy_threshold, schedule, new_dataset
    
    # Conditions
    drift_score_threshold: Mapped[float] = mapped_column(Float, default=0.25, nullable=False) # PSI threshold
    performance_drop_threshold: Mapped[float] = mapped_column(Float, default=0.05, nullable=False) # 5% drop in F1/Accuracy
    cron_schedule: Mapped[Optional[str]] = mapped_column(String(100), nullable=True) # e.g. "0 0 * * 0" (Weekly)
    
    # Target Configuration
    target_registered_model_id: Mapped[str] = mapped_column(String(36), ForeignKey("registered_models.id"), nullable=False)
    auto_promote_if_passed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    min_improvement_margin: Mapped[float] = mapped_column(Float, default=0.01, nullable=False) # New model must beat old by >= 1%

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    project = relationship("Project")
    target_model = relationship("RegisteredModel")
    executions: Mapped[List["RetrainingExecution"]] = relationship("RetrainingExecution", back_populates="policy", cascade="all, delete-orphan")


class RetrainingExecution(Base):
    __tablename__ = "retraining_executions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    policy_id: Mapped[str] = mapped_column(String(36), ForeignKey("retraining_policies.id", ondelete="CASCADE"), nullable=False)
    drift_event_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("drift_events.id"), nullable=True)
    
    status: Mapped[str] = mapped_column(String(50), default="running", nullable=False) # running, completed, champion_retained, challenger_promoted, failed
    trigger_reason: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # Runs & Comparison
    baseline_model_version_id: Mapped[str] = mapped_column(String(36), ForeignKey("model_versions.id"), nullable=False)
    new_candidate_model_version_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("model_versions.id"), nullable=True)
    
    baseline_metric_score: Mapped[float] = mapped_column(Float, nullable=False)
    challenger_metric_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    comparison_summary: Mapped[Optional[dict]] = mapped_column(JSON, default=dict, nullable=True)
    
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    policy: Mapped["RetrainingPolicy"] = relationship("RetrainingPolicy", back_populates="executions")
    drift_event = relationship("DriftEvent", back_populates="retraining_jobs")
    baseline_version = relationship("ModelVersion", foreign_keys=[baseline_model_version_id])
    challenger_version = relationship("ModelVersion", foreign_keys=[new_candidate_model_version_id])
