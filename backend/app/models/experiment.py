"""
ModelForge AI - Experiment Tracking & Run Models
Tracks hyperparameters, evaluation metrics, artifacts, system resource utilization, and git state.
"""

from datetime import datetime, timezone
import uuid
from typing import List, Optional
from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey, Integer, Float, String, Text, JSON
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from app.core.database import Base


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(150), index=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    tags: Mapped[Optional[list]] = mapped_column(JSON, default=list, nullable=True)
    is_archived: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

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
    project = relationship("Project", back_populates="experiments")
    creator = relationship("User")
    runs: Mapped[List["ExperimentRun"]] = relationship("ExperimentRun", back_populates="experiment", cascade="all, delete-orphan")


class ExperimentRun(Base):
    __tablename__ = "experiment_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    experiment_id: Mapped[str] = mapped_column(String(36), ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    
    dataset_version_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("dataset_versions.id"), nullable=True)
    algorithm_name: Mapped[str] = mapped_column(String(100), nullable=False) # xgboost, random_forest, lightgbm, pytorch_mlp, etc.
    framework: Mapped[str] = mapped_column(String(50), default="scikit-learn", nullable=False) # scikit-learn, xgboost, lightgbm, pytorch
    
    status: Mapped[str] = mapped_column(String(50), default="running", nullable=False) # queued, running, completed, failed, cancelled
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    hyperparameters: Mapped[Optional[dict]] = mapped_column(JSON, default=dict, nullable=True)
    metrics: Mapped[Optional[dict]] = mapped_column(JSON, default=dict, nullable=True) # accuracy, f1, precision, recall, roc_auc, etc.
    system_metrics: Mapped[Optional[dict]] = mapped_column(JSON, default=dict, nullable=True) # cpu_percent, memory_mb, gpu_utilization
    
    model_artifact_uri: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    code_version_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    error_traceback: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_by_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    experiment: Mapped["Experiment"] = relationship("Experiment", back_populates="runs")
    dataset_version = relationship("DatasetVersion")
    creator = relationship("User")
    registered_models = relationship("ModelVersion", back_populates="experiment_run")
