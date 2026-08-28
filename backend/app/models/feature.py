"""
ModelForge AI - Feature Store & Feature Lineage Models
Supports enterprise offline & online feature registration, transformations, and versioning.
"""

from datetime import datetime, timezone
import uuid
from typing import List, Optional
from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey, Integer, String, Text, JSON
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from app.core.database import Base


class Feature(Base):
    __tablename__ = "features"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    entity_name: Mapped[str] = mapped_column(String(100), default="default", nullable=False)  # e.g., "customer_id", "transaction_id"
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    data_type: Mapped[str] = mapped_column(String(50), nullable=False)  # float, int, string, boolean, timestamp, vector
    transformation_type: Mapped[str] = mapped_column(String(100), default="identity", nullable=False) # standard_scaler, one_hot, log, tfidf, custom
    transformation_params: Mapped[Optional[dict]] = mapped_column(JSON, default=dict, nullable=True)
    
    source_dataset_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("datasets.id"), nullable=True)
    source_column: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    status: Mapped[str] = mapped_column(String(50), default="active", nullable=False) # active, deprecated, archived
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
    project = relationship("Project", back_populates="features")
    creator = relationship("User")
    source_dataset = relationship("Dataset")
    versions: Mapped[List["FeatureVersion"]] = relationship("FeatureVersion", back_populates="feature", cascade="all, delete-orphan")


class FeatureVersion(Base):
    __tablename__ = "feature_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    feature_id: Mapped[str] = mapped_column(String(36), ForeignKey("features.id", ondelete="CASCADE"), nullable=False)
    version_tag: Mapped[str] = mapped_column(String(50), nullable=False)  # e.g., "v1.0.0"
    
    statistics: Mapped[Optional[dict]] = mapped_column(JSON, default=dict, nullable=True) # mean, std, quantiles, null_pct
    lineage_metadata: Mapped[Optional[dict]] = mapped_column(JSON, default=dict, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    feature: Mapped["Feature"] = relationship("Feature", back_populates="versions")


class FeatureView(Base):
    __tablename__ = "feature_views"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    entity_id: Mapped[str] = mapped_column(String(100), nullable=False)
    feature_names: Mapped[list] = mapped_column(JSON, nullable=False) # list of included feature names
    ttl_seconds: Mapped[int] = mapped_column(Integer, default=86400, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
