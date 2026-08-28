"""
ModelForge AI - Governance, Model Cards & Compliance API Endpoints
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, Query, Path
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db, get_current_user
from app.models.user import User
from app.schemas.common import APIResponse
from app.services.governance_service import GovernanceService
from app.services.cost_optimizer_service import CostOptimizerService

router = APIRouter(prefix="/governance", tags=["Governance & Model Cards"])


@router.get("/model-card/{model_version_id}", response_model=APIResponse[Dict[str, Any]])
async def get_model_card(
    model_version_id: str = Path(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    svc = GovernanceService(db)
    card = await svc.generate_model_card(model_version_id)
    return APIResponse(data=card, message="Model Card generated with cryptographic verification signature.")


@router.get("/cost-analysis/{deployment_id}", response_model=APIResponse[Dict[str, Any]])
async def get_deployment_cost(
    deployment_id: str = Path(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    svc = CostOptimizerService(db)
    analysis = await svc.get_deployment_cost_analysis(deployment_id)
    return APIResponse(data=analysis)
