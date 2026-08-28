"""
ModelForge AI - Dataset, Versioning, Schema, Quality & Profiling Models
"""

from datetime import datetime, timezone
import uuid
from typing import List, Optional
from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey, Integer, Float, String, Text, JSON, BigInteger
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from app.core.database import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(150), index=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    format: Mapped[str] = mapped_column(String(20), default="csv", nullable=False)  # csv, json, parquet, excel, sql
    target_column: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    source_type: Mapped[str] = mapped_column(String(50), default="upload", nullable=False)  # upload, s3, postgres, api
    source_uri: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    is_archived: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
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
    project = relationship("Project", back_populates="datasets")
    creator = relationship("User")
    versions: Mapped[List["DatasetVersion"]] = relationship("DatasetVersion", back_populates="dataset", cascade="all, delete-orphan")


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    dataset_id: Mapped[str] = mapped_column(String(36), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    version_tag: Mapped[str] = mapped_column(String(50), nullable=False)  # e.g., "v1.0.0"
    storage_uri: Mapped[str] = mapped_column(String(500), nullable=False)
    
    row_count: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    column_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    checksum_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    
    schema_definition: Mapped[Optional[dict]] = mapped_column(JSON, default=dict, nullable=True) # column names and inferred dtypes
    preview_data: Mapped[Optional[list]] = mapped_column(JSON, default=list, nullable=True)      # first 20 rows
    
    status: Mapped[str] = mapped_column(String(50), default="ready", nullable=False) # processing, ready, failed
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="versions")
    quality_reports: Mapped[List["DataQualityReport"]] = relationship("DataQualityReport", back_populates="dataset_version", cascade="all, delete-orphan")
    profile_reports: Mapped[List["DataProfileReport"]] = relationship("DataProfileReport", back_populates="dataset_version", cascade="all, delete-orphan")


class DataQualityReport(Base):
    __tablename__ = "data_quality_reports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    dataset_version_id: Mapped[str] = mapped_column(String(36), ForeignKey("dataset_versions.id", ondelete="CASCADE"), nullable=False)
    
    quality_score: Mapped[float] = mapped_column(Float, nullable=False)  # 0.0 to 100.0
    passed_rules: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failed_rules: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_checks: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    
    missing_value_percentage: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    duplicate_rows_count: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    outlier_count: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    
    anomalies: Mapped[Optional[dict]] = mapped_column(JSON, default=dict, nullable=True)
    rule_results: Mapped[Optional[list]] = mapped_column(JSON, default=list, nullable=True)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    dataset_version: Mapped["DatasetVersion"] = relationship("DatasetVersion", back_populates="quality_reports")


class DataProfileReport(Base):
    __tablename__ = "data_profile_reports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    dataset_version_id: Mapped[str] = mapped_column(String(36), ForeignKey("dataset_versions.id", ondelete="CASCADE"), nullable=False)
    
    column_stats: Mapped[dict] = mapped_column(JSON, nullable=False) # mean, median, std, min, max, quantiles per column
    correlations: Mapped[Optional[dict]] = mapped_column(JSON, default=dict, nullable=True) # Pearson/Spearman matrix
    histograms: Mapped[Optional[dict]] = mapped_column(JSON, default=dict, nullable=True)
    missingness_matrix: Mapped[Optional[dict]] = mapped_column(JSON, default=dict, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    dataset_version: Mapped["DatasetVersion"] = relationship("DatasetVersion", back_populates="profile_reports")
