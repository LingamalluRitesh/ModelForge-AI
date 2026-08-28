"""
ModelForge AI - Real-Time Inference & Batch Prediction Models
Tracks input features, output predictions, confidences, latencies, and batch jobs.
"""

from datetime import datetime, timezone
import uuid
from typing import List, Optional
from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey, Integer, Float, String, Text, JSON, BigInteger
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from app.core.database import Base


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    deployment_id: Mapped[str] = mapped_column(String(36), ForeignKey("deployments.id", ondelete="CASCADE"), index=True, nullable=False)
    model_version_id: Mapped[str] = mapped_column(String(36), ForeignKey("model_versions.id"), nullable=False)
    
    request_id: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    features: Mapped[dict] = mapped_column(JSON, nullable=False)
    prediction: Mapped[dict] = mapped_column(JSON, nullable=False)
    probability_or_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    status_code: Mapped[int] = mapped_column(Integer, default=200, nullable=False)
    
    # Ground truth feedback if provided later
    ground_truth: Mapped[Optional[dict]] = mapped_column(JSON, default=dict, nullable=True)
    feedback_timestamp: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True, nullable=False
    )

    deployment = relationship("Deployment", back_populates="predictions")
    model_version = relationship("ModelVersion")


class BatchInferenceJob(Base):
    __tablename__ = "batch_inference_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    model_version_id: Mapped[str] = mapped_column(String(36), ForeignKey("model_versions.id"), nullable=False)
    dataset_version_id: Mapped[str] = mapped_column(String(36), ForeignKey("dataset_versions.id"), nullable=False)
    
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="queued", nullable=False) # queued, processing, completed, failed
    
    total_rows: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    processed_rows: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    failed_rows: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    
    output_storage_uri: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_by_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    model_version = relationship("ModelVersion")
    dataset_version = relationship("DatasetVersion")
    creator = relationship("User")
