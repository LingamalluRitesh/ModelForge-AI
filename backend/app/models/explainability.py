"""
ModelForge AI - Explainable AI (SHAP) & Algorithmic Fairness Models
Provides Global / Local SHAP attribution and fairness metrics (Demographic Parity, Equalized Odds, Disparate Impact).
"""

from datetime import datetime, timezone
import uuid
from typing import List, Optional
from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey, Integer, Float, String, Text, JSON
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from app.core.database import Base


class SHAPExplanation(Base):
    __tablename__ = "shap_explanations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_version_id: Mapped[str] = mapped_column(String(36), ForeignKey("model_versions.id", ondelete="CASCADE"), nullable=False)
    explanation_type: Mapped[str] = mapped_column(String(50), default="global", nullable=False) # global, local_sample, partial_dependence
    
    # Global feature importances or Local prediction SHAP values
    feature_importances: Mapped[dict] = mapped_column(JSON, nullable=False) # { "credit_score": 0.35, "income": 0.28, ... }
    summary_plot_data: Mapped[Optional[dict]] = mapped_column(JSON, default=dict, nullable=True)
    sample_instances: Mapped[Optional[list]] = mapped_column(JSON, default=list, nullable=True) # local force plot vectors
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    model_version = relationship("ModelVersion")


class FairnessAnalysisReport(Base):
    __tablename__ = "fairness_analysis_reports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_version_id: Mapped[str] = mapped_column(String(36), ForeignKey("model_versions.id", ondelete="CASCADE"), nullable=False)
    sensitive_attribute: Mapped[str] = mapped_column(String(100), nullable=False) # e.g. "gender", "age_group", "race"
    
    demographic_parity_ratio: Mapped[float] = mapped_column(Float, nullable=False) # 4/5ths rule (0.80+)
    disparate_impact_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    equal_opportunity_difference: Mapped[float] = mapped_column(Float, nullable=False)
    equalized_odds_difference: Mapped[float] = mapped_column(Float, nullable=False)
    
    group_metrics: Mapped[dict] = mapped_column(JSON, nullable=False) # per-subgroup FPR, FNR, selection rate
    is_fair: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    recommendations: Mapped[Optional[list]] = mapped_column(JSON, default=list, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    model_version = relationship("ModelVersion")
