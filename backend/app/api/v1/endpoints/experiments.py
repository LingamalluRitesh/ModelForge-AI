"""
ModelForge AI - Experiments, Training & AutoML API Endpoints
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.deps import get_db, get_current_user
from app.models.user import User
from app.models.experiment import Experiment, ExperimentRun
from app.models.training import TrainingJob, AutoMLJob
from app.schemas.experiment import (
    ExperimentCreate, ExperimentResponse, ExperimentRunResponse,
    RunComparisonRequest, RunComparisonResponse
)
from app.schemas.training import (
    TrainingJobCreate, TrainingJobResponse, AutoMLJobCreate, AutoMLJobResponse
)
from app.schemas.common import APIResponse
from app.services.training_service import ExperimentService, TrainingService
from app.services.automl_service import AutoMLService

router = APIRouter(prefix="/experiments", tags=["Experiments & Training"])
train_router = APIRouter(prefix="/training", tags=["Model Training Engine"])
automl_router = APIRouter(prefix="/automl", tags=["AutoML Engine"])


# Experiments
@router.get("", response_model=APIResponse[List[ExperimentResponse]])
async def list_experiments(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List experiments in a project."""
    exp_svc = ExperimentService(db)
    experiments = await exp_svc.list_experiments(project_id)
    return APIResponse(data=[ExperimentResponse.model_validate(e) for e in experiments])


@router.post("", response_model=APIResponse[ExperimentResponse], status_code=status.HTTP_201_CREATED)
async def create_experiment(
    project_id: str,
    payload: ExperimentCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new experiment tracking workspace."""
    exp_svc = ExperimentService(db)
    exp = await exp_svc.create_experiment(project_id, current_user.id, payload)
    return APIResponse(data=ExperimentResponse.model_validate(exp), message="Experiment created successfully.")


@router.get("/{experiment_id}/runs", response_model=APIResponse[List[ExperimentRunResponse]])
async def list_runs(
    experiment_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all training runs inside an experiment."""
    query = select(ExperimentRun).where(ExperimentRun.experiment_id == experiment_id).order_by(ExperimentRun.created_at.desc())
    res = await db.execute(query)
    runs = list(res.scalars().all())
    return APIResponse(data=[ExperimentRunResponse.model_validate(r) for r in runs])


@router.post("/compare", response_model=APIResponse[RunComparisonResponse])
async def compare_runs(
    payload: RunComparisonRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Compare multiple experiment runs side-by-side."""
    exp_svc = ExperimentService(db)
    result = await exp_svc.compare_runs(payload.run_ids)
    return APIResponse(data=RunComparisonResponse(
        runs=[ExperimentRunResponse.model_validate(r) for r in result["runs"]],
        metric_keys=result["metric_keys"],
        hyperparameter_keys=result["hyperparameter_keys"],
        best_run_id_per_metric=result["best_run_id_per_metric"],
    ))


# Training
@train_router.post("/jobs", response_model=APIResponse[TrainingJobResponse], status_code=status.HTTP_201_CREATED)
async def launch_training_job(
    project_id: str,
    payload: TrainingJobCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Launch ML model training (Scikit-learn, XGBoost, LightGBM, PyTorch)."""
    train_svc = TrainingService(db)
    job, run = await train_svc.execute_training_job(project_id, current_user.id, payload)
    return APIResponse(data=TrainingJobResponse.model_validate(job), message="Training completed and model artifact created.")


# AutoML
@automl_router.post("/jobs", response_model=APIResponse[AutoMLJobResponse], status_code=status.HTTP_201_CREATED)
async def launch_automl_job(
    project_id: str,
    payload: AutoMLJobCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Launch end-to-end AutoML pipeline with Optuna Bayesian optimization."""
    automl_svc = AutoMLService(db)
    job = await automl_svc.launch_automl_job(project_id, current_user.id, payload)
    return APIResponse(data=AutoMLJobResponse.model_validate(job), message="AutoML exploration completed.")
