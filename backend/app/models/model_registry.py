"""
ModelForge AI - Model Registry, Governance, Quality Gates & Approval Models
Full enterprise lifecycle: Development -> Candidate -> Approved -> Staging -> Production -> Archived.
"""

from datetime import datetime, timezone
import uuid
from typing import List, Optional
from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey, Integer, Float, String, Text, JSON
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from app.core.database import Base


class ModelStage(str):
    DEVELOPMENT = "development"
    CANDIDATE = "candidate"
    VALIDATION = "validation"
    APPROVED = "approved"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class RegisteredModel(Base):
    __tablename__ = "registered_models"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(150), index=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    problem_type: Mapped[str] = mapped_column(String(50), nullable=False)
    
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
    project = relationship("Project", back_populates="models")
    creator = relationship("User")
    versions: Mapped[List["ModelVersion"]] = relationship("ModelVersion", back_populates="registered_model", cascade="all, delete-orphan")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    registered_model_id: Mapped[str] = mapped_column(String(36), ForeignKey("registered_models.id", ondelete="CASCADE"), nullable=False)
    version_tag: Mapped[str] = mapped_column(String(50), nullable=False)  # e.g., "v1.0.0", "v2.1.0"
    
    stage: Mapped[str] = mapped_column(String(50), default=ModelStage.DEVELOPMENT, nullable=False)
    experiment_run_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("experiment_runs.id"), nullable=True)
    
    algorithm_name: Mapped[str] = mapped_column(String(100), nullable=False)
    framework: Mapped[str] = mapped_column(String(50), nullable=False)
    
    storage_uri: Mapped[str] = mapped_column(String(500), nullable=False) # Model artifact pickle / onnx / pt
    signature: Mapped[Optional[dict]] = mapped_column(JSON, default=dict, nullable=True) # inputs & outputs schema
    metrics: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False) # validation & test metrics
    hyperparameters: Mapped[Optional[dict]] = mapped_column(JSON, default=dict, nullable=True)
    dependencies: Mapped[Optional[list]] = mapped_column(JSON, default=list, nullable=True)
    
    quality_gate_passed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    quality_gate_summary: Mapped[Optional[dict]] = mapped_column(JSON, default=dict, nullable=True)
    
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

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
    registered_model: Mapped["RegisteredModel"] = relationship("RegisteredModel", back_populates="versions")
    experiment_run = relationship("ExperimentRun", back_populates="registered_models")
    creator = relationship("User")
    approval_requests: Mapped[List["ModelApprovalRequest"]] = relationship("ModelApprovalRequest", back_populates="model_version", cascade="all, delete-orphan")
    deployments = relationship("Deployment", foreign_keys="[Deployment.model_version_id]", back_populates="model_version")


class ModelApprovalRequest(Base):
    __tablename__ = "model_approval_requests"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_version_id: Mapped[str] = mapped_column(String(36), ForeignKey("model_versions.id", ondelete="CASCADE"), nullable=False)
    target_stage: Mapped[str] = mapped_column(String(50), nullable=False) # staging, production
    
    status: Mapped[str] = mapped_column(String(50), default="pending", nullable=False) # pending, approved, rejected
    requested_by_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    reviewed_by_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("users.id"), nullable=True)
    
    review_comments: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    quality_checks_passed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    model_version: Mapped["ModelVersion"] = relationship("ModelVersion", back_populates="approval_requests")
    requester = relationship("User", foreign_keys=[requested_by_id])
    reviewer = relationship("User", foreign_keys=[reviewed_by_id])


class QualityGateConfig(Base):
    __tablename__ = "quality_gate_configs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    target_stage: Mapped[str] = mapped_column(String(50), default="production", nullable=False)
    
    rules: Mapped[list] = mapped_column(JSON, default=list, nullable=False) # e.g. [{"metric": "f1", "operator": ">=", "threshold": 0.85}, {"metric": "data_quality", "operator": ">=", "threshold": 90.0}]
    is_blocking: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
