"""
ModelForge AI - Dataset, Data Quality & Profiling Domain Service
Handles file ingestion, parsing, schema detection, automated quality evaluation, and statistical profiling.
"""

import io
import hashlib
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.exceptions import EntityNotFoundException, ValidationException
from app.core.storage import storage_engine
from app.models.dataset import Dataset, DatasetVersion, DataQualityReport, DataProfileReport
from app.schemas.dataset import DatasetCreate, DatasetUpdate, DataQualityRuleConfig
from ml_engine.data_quality.quality_engine import DataQualityEngine
from ml_engine.profiling.statistical_profiler import StatisticalProfiler


class DatasetService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_dataset_with_file(
        self,
        project_id: str,
        user_id: str,
        name: str,
        description: Optional[str],
        file_content: bytes,
        filename: str,
        target_column: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Tuple[Dataset, DatasetVersion]:
        """Ingest file, detect format & schema, compute checksum, save to storage, and create version."""
        # Detect extension
        ext = filename.split(".")[-1].lower() if "." in filename else "csv"
        if ext not in ("csv", "parquet", "json", "xlsx", "xls"):
            ext = "csv"

        # Parse DataFrame to validate structure
        try:
            if ext == "csv":
                df = pd.read_csv(io.BytesIO(file_content))
            elif ext == "parquet":
                df = pd.read_parquet(io.BytesIO(file_content))
            elif ext == "json":
                df = pd.read_json(io.BytesIO(file_content))
            elif ext in ("xlsx", "xls"):
                df = pd.read_excel(io.BytesIO(file_content))
            else:
                df = pd.read_csv(io.BytesIO(file_content))
        except Exception as e:
            raise ValidationException(f"Failed to parse uploaded dataset file: {str(e)}")

        checksum = hashlib.sha256(file_content).hexdigest()

        # Inferred schema definition
        schema_def = {
            col: str(df[col].dtype)
            for col in df.columns
        }

        # Preview rows (first 10)
        preview_data = df.head(10).replace({np.nan: None}).to_dict(orient="records")

        # Create parent Dataset record
        dataset = Dataset(
            project_id=project_id,
            name=name,
            description=description,
            format=ext,
            target_column=target_column,
            tags=tags or [],
            created_by_id=user_id,
        )
        self.session.add(dataset)
        await self.session.flush()

        # Save to object storage
        storage_key = f"projects/{project_id}/datasets/{dataset.id}/v1.0.0.{ext}"
        storage_uri = await storage_engine.save_file(file_content, storage_key)

        # Create Version 1.0.0
        version = DatasetVersion(
            dataset_id=dataset.id,
            version_tag="v1.0.0",
            storage_uri=storage_uri,
            row_count=len(df),
            column_count=len(df.columns),
            file_size_bytes=len(file_content),
            checksum_sha256=checksum,
            schema_definition=schema_def,
            preview_data=preview_data,
            status="ready",
        )
        self.session.add(version)
        await self.session.flush()

        # Automatically execute initial Data Quality Analysis & Profiling
        dq_engine = DataQualityEngine()
        dq_results = dq_engine.evaluate_quality(df)

        dq_report = DataQualityReport(
            dataset_version_id=version.id,
            quality_score=dq_results["quality_score"],
            passed_rules=dq_results["passed_rules"],
            failed_rules=dq_results["failed_rules"],
            total_checks=dq_results["total_checks"],
            missing_value_percentage=dq_results["missing_value_percentage"],
            duplicate_rows_count=dq_results["duplicate_rows_count"],
            outlier_count=dq_results["outlier_count"],
            anomalies=dq_results["anomalies"],
            rule_results=dq_results["rule_results"],
            summary=dq_results["summary"],
        )
        self.session.add(dq_report)

        profile_results = StatisticalProfiler.profile_dataset(df)
        profile_report = DataProfileReport(
            dataset_version_id=version.id,
            column_stats=profile_results["column_stats"],
            correlations=profile_results["correlations"],
            histograms=profile_results["histograms"],
            missingness_matrix=profile_results["missingness_matrix"],
        )
        self.session.add(profile_report)

        await self.session.commit()
        return dataset, version

    async def get_dataframe_for_version(self, version_id: str) -> pd.DataFrame:
        """Load DataFrame into memory from storage URI."""
        query = select(DatasetVersion).where(DatasetVersion.id == version_id)
        res = await self.session.execute(query)
        ver = res.scalar_one_or_none()
        if not ver:
            raise EntityNotFoundException("DatasetVersion", version_id)

        file_bytes = await storage_engine.read_file(ver.storage_uri)
        if ver.storage_uri.endswith(".parquet"):
            return pd.read_parquet(io.BytesIO(file_bytes))
        elif ver.storage_uri.endswith(".json"):
            return pd.read_json(io.BytesIO(file_bytes))
        else:
            return pd.read_csv(io.BytesIO(file_bytes))

    async def run_data_quality(self, version_id: str, custom_rules: Optional[List[DataQualityRuleConfig]] = None) -> DataQualityReport:
        """Run custom rule-based data quality analysis on dataset version."""
        df = await self.get_dataframe_for_version(version_id)
        rules_dicts = [r.model_dump() for r in custom_rules] if custom_rules else []
        dq_engine = DataQualityEngine(custom_rules=rules_dicts)
        dq_results = dq_engine.evaluate_quality(df)

        report = DataQualityReport(
            dataset_version_id=version_id,
            quality_score=dq_results["quality_score"],
            passed_rules=dq_results["passed_rules"],
            failed_rules=dq_results["failed_rules"],
            total_checks=dq_results["total_checks"],
            missing_value_percentage=dq_results["missing_value_percentage"],
            duplicate_rows_count=dq_results["duplicate_rows_count"],
            outlier_count=dq_results["outlier_count"],
            anomalies=dq_results["anomalies"],
            rule_results=dq_results["rule_results"],
            summary=dq_results["summary"],
        )
        self.session.add(report)
        await self.session.commit()
        return report
