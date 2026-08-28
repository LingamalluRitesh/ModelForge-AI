"""
ModelForge AI - Retraining API Endpoints
"""

from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from app.services.retraining_trigger_service import RetrainingTriggerService

router = APIRouter()


class EvaluateDriftTriggerRequest(BaseModel):
    model_version_id: str
    current_psi_score: float = Field(..., ge=0.0)
    current_error_rate: float = Field(..., ge=0.0, le=1.0)
    policy_config: Optional[Dict[str, Any]] = None


@router.post("/evaluate-trigger")
async def evaluate_retraining_trigger(request: EvaluateDriftTriggerRequest):
    """Evaluate drift metrics and launch automated retraining pipeline if SLA breached."""
    try:
        res = RetrainingTriggerService.evaluate_and_trigger(
            model_version_id=request.model_version_id,
            current_psi_score=request.current_psi_score,
            current_error_rate=request.current_error_rate,
            policy_config=request.policy_config,
        )
        return res
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retraining trigger evaluation failed: {str(e)}",
        )
