"""
ModelForge AI - Data Drift & Concept Drift Models
Tracks statistical divergence across features: PSI, KS-test, Chi-square, Wasserstein, Jensen-Shannon.
"""

from datetime import datetime, timezone
import uuid
from typing import List, Optional
from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey, Integer, Float, String, Text, JSON
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from app.core.database import Base


class DriftEvent(Base):
    __tablename__ = "drift_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    deployment_id: Mapped[str] = mapped_column(String(36), ForeignKey("deployments.id", ondelete="CASCADE"), index=True, nullable=False)
    
    drift_type: Mapped[str] = mapped_column(String(50), default="data_drift", nullable=False) # data_drift, concept_drift, label_drift
    severity: Mapped[str] = mapped_column(String(20), default="medium", nullable=False) # low, medium, high, critical
    
    overall_drift_score: Mapped[float] = mapped_column(Float, nullable=False) # Normalized 0.0 to 1.0
    drifted_features_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_features_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    
    baseline_dataset_version_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("dataset_versions.id"), nullable=True)
    target_sample_size: Mapped[int] = mapped_column(Integer, nullable=False)
    
    feature_metrics: Mapped[dict] = mapped_column(JSON, nullable=False) # Per-feature PSI, KS, Chi2, p-values
    recommendations: Mapped[Optional[list]] = mapped_column(JSON, default=list, nullable=True)
    
    is_resolved: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    retraining_triggered: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True, nullable=False
    )

    deployment = relationship("Deployment", back_populates="drift_events")
    baseline_dataset = relationship("DatasetVersion")
    retraining_jobs = relationship("RetrainingExecution", back_populates="drift_event")
