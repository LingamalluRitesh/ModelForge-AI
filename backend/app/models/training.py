"""
ModelForge AI - Training Job, AutoML & Hyperparameter Optimization Models
"""

from datetime import datetime, timezone
import uuid
from typing import List, Optional
from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey, Integer, Float, String, Text, JSON
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from app.core.database import Base


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    experiment_id: Mapped[str] = mapped_column(String(36), ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False)
    
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    algorithm: Mapped[str] = mapped_column(String(100), nullable=False)
    dataset_version_id: Mapped[str] = mapped_column(String(36), ForeignKey("dataset_versions.id"), nullable=False)
    
    target_column: Mapped[str] = mapped_column(String(100), nullable=False)
    feature_columns: Mapped[list] = mapped_column(JSON, nullable=False)
    hyperparameters: Mapped[Optional[dict]] = mapped_column(JSON, default=dict, nullable=True)
    
    status: Mapped[str] = mapped_column(String(50), default="queued", nullable=False) # queued, running, completed, failed, cancelled
    progress_percentage: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    current_epoch_or_step: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    
    metrics: Mapped[Optional[dict]] = mapped_column(JSON, default=dict, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    run_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("experiment_runs.id"), nullable=True)

    created_by_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class AutoMLJob(Base):
    __tablename__ = "automl_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    experiment_id: Mapped[str] = mapped_column(String(36), ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False)
    
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    dataset_version_id: Mapped[str] = mapped_column(String(36), ForeignKey("dataset_versions.id"), nullable=False)
    target_column: Mapped[str] = mapped_column(String(100), nullable=False)
    problem_type: Mapped[str] = mapped_column(String(50), nullable=False) # binary_classification, multiclass_classification, regression
    
    optimization_metric: Mapped[str] = mapped_column(String(50), default="f1", nullable=False) # f1, accuracy, roc_auc, rmse, r2
    time_limit_seconds: Mapped[int] = mapped_column(Integer, default=3600, nullable=False)
    max_trials: Mapped[int] = mapped_column(Integer, default=20, nullable=False)
    algorithms_to_evaluate: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    
    status: Mapped[str] = mapped_column(String(50), default="queued", nullable=False)
    best_algorithm: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    best_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    best_run_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("experiment_runs.id"), nullable=True)
    
    leaderboard: Mapped[Optional[list]] = mapped_column(JSON, default=list, nullable=True) # list of trial summaries ranked by metric
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_by_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    trials: Mapped[List["AutoMLTrial"]] = relationship("AutoMLTrial", back_populates="automl_job", cascade="all, delete-orphan")


class AutoMLTrial(Base):
    __tablename__ = "automl_trials"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    automl_job_id: Mapped[str] = mapped_column(String(36), ForeignKey("automl_jobs.id", ondelete="CASCADE"), nullable=False)
    trial_number: Mapped[int] = mapped_column(Integer, nullable=False)
    algorithm_name: Mapped[str] = mapped_column(String(100), nullable=False)
    
    hyperparameters: Mapped[dict] = mapped_column(JSON, nullable=False)
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False)
    primary_metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="completed", nullable=False) # completed, failed, pruned
    run_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("experiment_runs.id"), nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    automl_job: Mapped["AutoMLJob"] = relationship("AutoMLJob", back_populates="trials")
