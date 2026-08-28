"""
ModelForge AI - Model Monitoring & System Observability Models
Tracks real-time throughput, latency percentiles (p50, p95, p99), error rates,
accuracy/F1 scores from feedback, and hardware utilization.
"""

from datetime import datetime, timezone
import uuid
from typing import List, Optional
from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey, Integer, Float, String, Text, JSON, BigInteger
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from app.core.database import Base


class ModelMonitoringMetric(Base):
    __tablename__ = "model_monitoring_metrics"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    deployment_id: Mapped[str] = mapped_column(String(36), ForeignKey("deployments.id", ondelete="CASCADE"), index=True, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True, nullable=False)
    
    # Application & Throughput Metrics
    request_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_rate: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    
    # Latency Percentiles (ms)
    latency_p50: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    latency_p95: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    latency_p99: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    
    # ML Performance Metrics (computed when feedback is present)
    observed_accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    observed_f1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    observed_precision: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    observed_recall: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Statistical Summary of Predictions in this Window
    prediction_distribution: Mapped[Optional[dict]] = mapped_column(JSON, default=dict, nullable=True)
    confidence_distribution: Mapped[Optional[dict]] = mapped_column(JSON, default=dict, nullable=True)

    deployment = relationship("Deployment", back_populates="monitoring_metrics")


class SystemResourceSnapshot(Base):
    __tablename__ = "system_resource_snapshots"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True, nullable=False)
    
    cpu_percent: Mapped[float] = mapped_column(Float, nullable=False)
    memory_used_mb: Mapped[float] = mapped_column(Float, nullable=False)
    memory_total_mb: Mapped[float] = mapped_column(Float, nullable=False)
    memory_percent: Mapped[float] = mapped_column(Float, nullable=False)
    disk_used_gb: Mapped[float] = mapped_column(Float, nullable=False)
    disk_total_gb: Mapped[float] = mapped_column(Float, nullable=False)
    
    active_worker_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    queued_task_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
