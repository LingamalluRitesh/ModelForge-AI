"""
ModelForge AI - Model Registry, Governance, Deployments & Serving Endpoints
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, Query, Path, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.api.deps import get_db, get_current_user
from app.models.user import User
from app.models.model_registry import RegisteredModel, ModelVersion, ModelApprovalRequest
from app.models.deployment import Deployment
from app.schemas.model_registry import (
    RegisteredModelCreate, RegisteredModelResponse,
    ModelVersionCreate, ModelVersionResponse,
    ModelApprovalRequestCreate, ModelApprovalRequestResponse,
    ModelApprovalVote, QualityGateConfigSchema
)
from app.schemas.deployment import (
    DeploymentCreate, DeploymentUpdate, DeploymentResponse,
    CanaryTrafficUpdateRequest, RollbackRequest, RollbackHistoryResponse
)
from app.schemas.common import APIResponse
from app.services.model_registry_service import ModelRegistryService
from app.services.deployment_service import DeploymentService
from ml_engine.inference.realtime_server import RealtimeInferenceEngine

reg_router = APIRouter(prefix="/registry", tags=["Model Registry & Governance"])
deploy_router = APIRouter(prefix="/deployments", tags=["Deployments & Serving"])
pred_router = APIRouter(prefix="/predict", tags=["Real-Time Inference"])


# Model Registry
@reg_router.get("/models", response_model=APIResponse[List[RegisteredModelResponse]])
async def list_registered_models(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all registered models in a project with versions."""
    query = select(RegisteredModel).options(selectinload(RegisteredModel.versions)).where(RegisteredModel.project_id == project_id, RegisteredModel.is_archived == False)
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
    return APIResponse(
        data=RegisteredModelResponse(
            id=model.id,
            project_id=model.project_id,
            name=model.name,
            description=model.description,
            problem_type=model.problem_type,
            tags=model.tags or [],
            is_archived=model.is_archived,
            created_at=model.created_at,
            updated_at=model.updated_at,
            versions=[],
        ),
        message="Registered model created.",
    )


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


@reg_router.post("/versions/{version_id}/request-approval", response_model=APIResponse[ModelApprovalRequestResponse], status_code=status.HTTP_201_CREATED)
async def request_model_approval(
    version_id: str,
    payload: ModelApprovalRequestCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Request governance approval to promote model version."""
    reg_svc = ModelRegistryService(db)
    approval = await reg_svc.request_approval(version_id, current_user.id, payload)
    return APIResponse(data=ModelApprovalRequestResponse.model_validate(approval), message="Model approval requested.")


@reg_router.post("/approvals/{approval_id}/review", response_model=APIResponse[ModelApprovalRequestResponse])
async def review_model_approval(
    approval_id: str,
    payload: ModelApprovalVote,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Approve or reject a model promotion."""
    reg_svc = ModelRegistryService(db)
    approval = await reg_svc.review_approval(approval_id, current_user.id, payload)
    return APIResponse(data=ModelApprovalRequestResponse.model_validate(approval), message=f"Model approval {approval.status}.")


# Deployments
@deploy_router.get("", response_model=APIResponse[List[DeploymentResponse]])
async def list_deployments(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all serving endpoints and deployments for a project."""
    deploy_svc = DeploymentService(db)
    deps = await deploy_svc.list_deployments(project_id)
    return APIResponse(data=[DeploymentResponse.model_validate(d) for d in deps])


@deploy_router.post("", response_model=APIResponse[DeploymentResponse], status_code=status.HTTP_201_CREATED)
async def create_deployment(
    project_id: str,
    payload: DeploymentCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Deploy model version with Canary, Blue/Green, or Direct rollout strategy."""
    deploy_svc = DeploymentService(db)
    dep = await deploy_svc.create_deployment(project_id, current_user.id, payload)
    return APIResponse(data=DeploymentResponse.model_validate(dep), message="Deployment initialized successfully.")


@deploy_router.post("/{deployment_id}/canary", response_model=APIResponse[DeploymentResponse])
async def update_canary_split(
    deployment_id: str,
    payload: CanaryTrafficUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Adjust live Canary traffic percentage split."""
    deploy_svc = DeploymentService(db)
    dep = await deploy_svc.update_canary_traffic(deployment_id, payload.canary_stage_percentage)
    return APIResponse(data=DeploymentResponse.model_validate(dep), message="Canary traffic split updated.")


@deploy_router.post("/{deployment_id}/rollback", response_model=APIResponse[DeploymentResponse])
async def rollback_deployment(
    deployment_id: str,
    payload: RollbackRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Execute immediate zero-downtime rollback to previous healthy version."""
    deploy_svc = DeploymentService(db)
    dep = await deploy_svc.rollback_deployment(deployment_id, current_user.id, payload.reason)
    return APIResponse(data=DeploymentResponse.model_validate(dep), message="Rollback executed successfully.")


# Real-time Prediction Engine
@pred_router.post("/{endpoint_path}")
async def serve_prediction(
    endpoint_path: str,
    payload: Dict[str, Any],
    db: AsyncSession = Depends(get_db),
):
    """Ultra-low latency real-time inference serving endpoint."""
    engine = RealtimeInferenceEngine(db)
    return await engine.predict(endpoint_path, payload)
