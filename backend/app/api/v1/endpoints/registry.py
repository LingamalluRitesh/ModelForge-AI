"""
ModelForge AI - Model Registry, Governance, Deployments & Predictions API Endpoints
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.deps import get_db, get_current_user
from app.models.user import User
from app.models.model_registry import RegisteredModel, ModelVersion, ModelApprovalRequest
from app.models.deployment import Deployment
from app.schemas.model_registry import (
    RegisteredModelCreate, RegisteredModelResponse, ModelVersionCreate, ModelVersionResponse,
    ModelApprovalRequestCreate, ModelApprovalVote, ModelApprovalRequestResponse
)
from app.schemas.deployment import (
    DeploymentCreate, DeploymentUpdate, DeploymentResponse, CanaryUpdateRequest,
    ABTestTrafficUpdateRequest, RollbackRequest
)
from app.schemas.prediction import RealtimePredictionRequest, RealtimePredictionResponse
from app.schemas.common import APIResponse
from app.services.model_registry_service import ModelRegistryService
from app.services.deployment_service import DeploymentService, InferenceService

reg_router = APIRouter(prefix="/registry", tags=["Model Registry & Governance"])
deploy_router = APIRouter(prefix="/deployments", tags=["Model Deployments & Traffic Routing"])
pred_router = APIRouter(prefix="/predictions", tags=["Inference & Real-Time Serving"])


# Model Registry
@reg_router.get("/models", response_model=APIResponse[List[RegisteredModelResponse]])
async def list_registered_models(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all registered models in a project with versions."""
    query = select(RegisteredModel).where(RegisteredModel.project_id == project_id, RegisteredModel.is_archived == False)
    res = await db.execute(query)
    models = list(res.scalars().all())
    return APIResponse(data=[RegisteredModelResponse.model_validate(m) for m in models])


@reg_router.post("/models", response_model=APIResponse[RegisteredModelResponse], status_code=status.HTTP_201_CREATED)
async def create_registered_model(
    project_id: str,
    payload: RegisteredModelCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new model registry entry."""
    reg_svc = ModelRegistryService(db)
    model = await reg_svc.create_registered_model(project_id, current_user.id, payload)
    return APIResponse(data=RegisteredModelResponse.model_validate(model), message="Registered model created.")


@reg_router.post("/models/{model_id}/versions", response_model=APIResponse[ModelVersionResponse], status_code=status.HTTP_201_CREATED)
async def register_version(
    model_id: str,
    payload: ModelVersionCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Register a new version from a trained experiment run."""
    reg_svc = ModelRegistryService(db)
    version = await reg_svc.register_model_version(model_id, current_user.id, payload)
    return APIResponse(data=ModelVersionResponse.model_validate(version), message="Model version registered successfully.")


@reg_router.post("/versions/{version_id}/request-approval", response_model=APIResponse[ModelApprovalRequestResponse])
async def request_model_approval(
    version_id: str,
    payload: ModelApprovalRequestCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Request governance approval to promote model to staging or production."""
    reg_svc = ModelRegistryService(db)
    req = await reg_svc.request_approval(version_id, current_user.id, payload)
    return APIResponse(data=ModelApprovalRequestResponse.model_validate(req), message="Approval request submitted.")


@reg_router.post("/approvals/{approval_id}/review", response_model=APIResponse[ModelApprovalRequestResponse])
async def review_model_approval(
    approval_id: str,
    payload: ModelApprovalVote,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Reviewer approves or rejects a model promotion."""
    reg_svc = ModelRegistryService(db)
    req = await reg_svc.review_approval(approval_id, current_user.id, payload)
    return APIResponse(data=ModelApprovalRequestResponse.model_validate(req), message=f"Model approval {payload.action}d.")


# Deployments
@deploy_router.get("", response_model=APIResponse[List[DeploymentResponse]])
async def list_deployments(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all deployed model endpoints."""
    deploy_svc = DeploymentService(db)
    deployments = await deploy_svc.list_deployments(project_id)
    return APIResponse(data=[DeploymentResponse.model_validate(d) for d in deployments])


@deploy_router.post("", response_model=APIResponse[DeploymentResponse], status_code=status.HTTP_201_CREATED)
async def create_deployment(
    project_id: str,
    payload: DeploymentCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Deploy model to Staging or Production."""
    deploy_svc = DeploymentService(db)
    deployment = await deploy_svc.create_deployment(project_id, current_user.id, payload)
    return APIResponse(data=DeploymentResponse.model_validate(deployment), message="Model deployed successfully.")


@deploy_router.post("/{deployment_id}/canary", response_model=APIResponse[DeploymentResponse])
async def update_canary(
    deployment_id: str,
    payload: CanaryUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update Canary rollout percentage."""
    deploy_svc = DeploymentService(db)
    dep = await deploy_svc.update_canary_stage(deployment_id, payload)
    return APIResponse(data=DeploymentResponse.model_validate(dep), message="Canary stage updated.")


@deploy_router.post("/{deployment_id}/rollback", response_model=APIResponse[DeploymentResponse])
async def rollback_deployment(
    deployment_id: str,
    payload: RollbackRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Roll back deployment to a previous healthy model version."""
    deploy_svc = DeploymentService(db)
    dep = await deploy_svc.execute_rollback(deployment_id, payload)
    return APIResponse(data=DeploymentResponse.model_validate(dep), message="Deployment rolled back successfully.")


# Predictions
@pred_router.post("/{endpoint_path}", response_model=RealtimePredictionResponse)
async def predict(
    endpoint_path: str,
    payload: RealtimePredictionRequest,
    db: AsyncSession = Depends(get_db),
):
    """Execute high-speed real-time prediction against deployed endpoint."""
    inf_svc = InferenceService(db)
    return await inf_svc.predict_realtime(endpoint_path, payload)
