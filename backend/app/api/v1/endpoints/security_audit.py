"""
ModelForge AI - Model Security & Adversarial Vulnerability Audit API Endpoints
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db, get_current_user
from app.models.user import User
from app.schemas.common import APIResponse
from app.services.security_service import ModelSecurityService

router = APIRouter(prefix="/security-audit", tags=["Security & Robustness"])


class SecurityAuditRequest(BaseModel):
    model_version_id: str
    test_dataset_version_id: str
    target_column: str
    epsilon: float = 0.05


@router.post("/adversarial-test", response_model=APIResponse[Dict[str, Any]])
async def run_adversarial_test(
    payload: SecurityAuditRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    svc = ModelSecurityService(db)
    res = await svc.audit_adversarial_robustness(
        model_version_id=payload.model_version_id,
        test_dataset_version_id=payload.test_dataset_version_id,
        target_column=payload.target_column,
        epsilon=payload.epsilon,
    )
    return APIResponse(data=res, message="Adversarial penetration test completed.")
