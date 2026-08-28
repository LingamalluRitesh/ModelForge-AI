"""
ModelForge AI - Dataset, Data Quality & Profiling Endpoints
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.deps import get_db, get_current_user
from app.models.user import User
from app.models.dataset import Dataset, DatasetVersion, DataQualityReport, DataProfileReport
from app.schemas.dataset import (
    DatasetResponse, DatasetVersionResponse, DataQualityReportResponse, DataProfileReportResponse,
    DataQualityAnalysisRequest
)
from app.schemas.common import APIResponse
from app.services.dataset_service import DatasetService

router = APIRouter(prefix="/datasets", tags=["Data Management & Quality"])


@router.get("", response_model=APIResponse[List[DatasetResponse]])
async def list_datasets(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List datasets in an ML project."""
    query = select(Dataset).where(Dataset.project_id == project_id, Dataset.is_archived == False)
    res = await db.execute(query)
    datasets = list(res.scalars().all())

    output = []
    for d in datasets:
        resp = DatasetResponse.model_validate(d)
        if d.versions:
            resp.latest_version = DatasetVersionResponse.model_validate(d.versions[-1])
        output.append(resp)

    return APIResponse(data=output)


@router.post("/upload", response_model=APIResponse[DatasetResponse], status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    project_id: str = Form(...),
    name: str = Form(...),
    description: Optional[str] = Form(None),
    target_column: Optional[str] = Form(None),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Upload and ingest a dataset file (CSV, Parquet, JSON, Excel), auto-compute Quality & Profile."""
    content = await file.read()
    dataset_svc = DatasetService(db)
    dataset, version = await dataset_svc.create_dataset_with_file(
        project_id=project_id,
        user_id=current_user.id,
        name=name,
        description=description,
        file_content=content,
        filename=file.filename or "dataset.csv",
        target_column=target_column,
    )

    resp = DatasetResponse.model_validate(dataset)
    resp.latest_version = DatasetVersionResponse.model_validate(version)
    return APIResponse(data=resp, message="Dataset ingested, validated, and profiled successfully.")


@router.get("/versions/{version_id}/quality", response_model=APIResponse[DataQualityReportResponse])
async def get_data_quality_report(
    version_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get the automated data quality score and rule validation report for a dataset version."""
    query = select(DataQualityReport).where(DataQualityReport.dataset_version_id == version_id).order_by(DataQualityReport.created_at.desc())
    res = await db.execute(query)
    report = res.scalars().first()
    if not report:
        # Run on the fly
        dataset_svc = DatasetService(db)
        report = await dataset_svc.run_data_quality(version_id)

    return APIResponse(data=DataQualityReportResponse.model_validate(report))


@router.get("/versions/{version_id}/profile", response_model=APIResponse[DataProfileReportResponse])
async def get_data_profile_report(
    version_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get statistical column distributions, quantiles, and correlation matrices."""
    query = select(DataProfileReport).where(DataProfileReport.dataset_version_id == version_id).order_by(DataProfileReport.created_at.desc())
    res = await db.execute(query)
    report = res.scalars().first()
    if not report:
        dataset_svc = DatasetService(db)
        df = await dataset_svc.get_dataframe_for_version(version_id)
        from app.ml_engine.profiling.statistical_profiler import StatisticalProfiler
        p_res = StatisticalProfiler.profile_dataset(df)
        report = DataProfileReport(
            dataset_version_id=version_id,
            column_stats=p_res["column_stats"],
            correlations=p_res["correlations"],
            histograms=p_res["histograms"],
            missingness_matrix=p_res["missingness_matrix"],
        )
        db.add(report)
        await db.commit()

    return APIResponse(data=DataProfileReportResponse.model_validate(report))
