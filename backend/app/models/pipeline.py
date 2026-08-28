"""
ModelForge AI - Visual ML Pipeline DAG & Workflow Scheduler Models
Supports visual pipeline creation (Dataset -> Validation -> FeatureEng -> Train -> Evaluate -> Approve -> Deploy),
topological execution, and cron/event schedules.
"""

from datetime import datetime, timezone
import uuid
from typing import List, Optional
from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey, Integer, Float, String, Text, JSON
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from app.core.database import Base


class MLPipeline(Base):
    __tablename__ = "ml_pipelines"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(150), index=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Visual DAG definition (nodes & edges representation)
    dag_definition: Mapped[dict] = mapped_column(JSON, nullable=False) # { "nodes": [...], "edges": [...] }
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    tags: Mapped[Optional[list]] = mapped_column(JSON, default=list, nullable=True)

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
    project = relationship("Project", back_populates="pipelines")
    creator = relationship("User")
    runs: Mapped[List["PipelineRun"]] = relationship("PipelineRun", back_populates="pipeline", cascade="all, delete-orphan")
    schedules: Mapped[List["WorkflowSchedule"]] = relationship("WorkflowSchedule", back_populates="pipeline", cascade="all, delete-orphan")


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    pipeline_id: Mapped[str] = mapped_column(String(36), ForeignKey("ml_pipelines.id", ondelete="CASCADE"), nullable=False)
    run_number: Mapped[int] = mapped_column(Integer, nullable=False)
    
    status: Mapped[str] = mapped_column(String(50), default="running", nullable=False) # queued, running, completed, failed, cancelled
    trigger_type: Mapped[str] = mapped_column(String(50), default="manual", nullable=False) # manual, schedule, drift_event, webhook
    
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    execution_state: Mapped[Optional[dict]] = mapped_column(JSON, default=dict, nullable=True) # node outputs & intermediate state
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    pipeline: Mapped["MLPipeline"] = relationship("MLPipeline", back_populates="runs")
    node_executions: Mapped[List["PipelineNodeExecution"]] = relationship("PipelineNodeExecution", back_populates="pipeline_run", cascade="all, delete-orphan")


class PipelineNodeExecution(Base):
    __tablename__ = "pipeline_node_executions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    pipeline_run_id: Mapped[str] = mapped_column(String(36), ForeignKey("pipeline_runs.id", ondelete="CASCADE"), nullable=False)
    node_id: Mapped[str] = mapped_column(String(100), nullable=False)
    node_type: Mapped[str] = mapped_column(String(50), nullable=False) # dataset, validation, feature_eng, train, evaluate, approve, deploy, notify
    
    status: Mapped[str] = mapped_column(String(50), default="running", nullable=False) # running, completed, failed, skipped
    inputs: Mapped[Optional[dict]] = mapped_column(JSON, default=dict, nullable=True)
    outputs: Mapped[Optional[dict]] = mapped_column(JSON, default=dict, nullable=True)
    logs: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    pipeline_run: Mapped["PipelineRun"] = relationship("PipelineRun", back_populates="node_executions")


class WorkflowSchedule(Base):
    __tablename__ = "workflow_schedules"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    pipeline_id: Mapped[str] = mapped_column(String(36), ForeignKey("ml_pipelines.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    
    cron_expression: Mapped[str] = mapped_column(String(100), nullable=False) # e.g. "0 0 * * *" (Daily midnight)
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    
    last_run_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    next_run_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    pipeline: Mapped["MLPipeline"] = relationship("MLPipeline", back_populates="schedules")
