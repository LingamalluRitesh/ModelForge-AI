"""
ModelForge AI - Dataset, Quality & Profiling Schemas
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class DatasetCreate(BaseModel):
    name: str
    description: Optional[str] = None
    format: str = "csv"  # csv, json, parquet, excel
    target_column: Optional[str] = None
    tags: Optional[List[str]] = None


class DatasetUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    target_column: Optional[str] = None
    tags: Optional[List[str]] = None
    is_archived: Optional[bool] = None


class DataQualityRuleConfig(BaseModel):
    column_name: str
    rule_type: str # not_null, range, unique, regex, data_type, value_in_set
    params: Dict[str, Any] = Field(default_factory=dict)
    severity: str = "error" # error, warning


class DataQualityAnalysisRequest(BaseModel):
    rules: Optional[List[DataQualityRuleConfig]] = None


class DataQualityReportResponse(BaseModel):
    id: str
    dataset_version_id: str
    quality_score: float
    passed_rules: int
    failed_rules: int
    total_checks: int
    missing_value_percentage: float
    duplicate_rows_count: int
    outlier_count: int
    anomalies: Optional[Dict[str, Any]] = None
    rule_results: Optional[List[Dict[str, Any]]] = None
    summary: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class DataProfileReportResponse(BaseModel):
    id: str
    dataset_version_id: str
    column_stats: Dict[str, Any]
    correlations: Optional[Dict[str, Any]] = None
    histograms: Optional[Dict[str, Any]] = None
    missingness_matrix: Optional[Dict[str, Any]] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class DatasetVersionResponse(BaseModel):
    id: str
    dataset_id: str
    version_tag: str
    storage_uri: str
    row_count: int
    column_count: int
    file_size_bytes: int
    checksum_sha256: Optional[str] = None
    schema_definition: Optional[Dict[str, Any]] = None
    preview_data: Optional[List[Dict[str, Any]]] = None
    status: str
    error_message: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class DatasetResponse(BaseModel):
    id: str
    project_id: str
    name: str
    description: Optional[str] = None
    format: str
    target_column: Optional[str] = None
    source_type: str
    is_archived: bool
    tags: Optional[List[str]] = None
    created_at: datetime
    updated_at: datetime
    latest_version: Optional[DatasetVersionResponse] = None

    model_config = {"from_attributes": True}
