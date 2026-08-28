"""
ModelForge AI - Causal Inference API Endpoints
"""

from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from app.services.causal_analysis_service import CausalAnalysisService

router = APIRouter()


class CausalEstimateRequest(BaseModel):
    records: List[Dict[str, Any]]
    treatment_column: str
    outcome_column: str
    confounders: List[str]
    method: str = "dml"


class CausalEstimateResponse(BaseModel):
    method: str
    treatment_column: str
    outcome_column: str
    confounders: List[str]
    estimated_ate: float
    standard_error: float
    t_statistic: float
    ci_lower: float
    ci_upper: float
    statistically_significant: bool


@router.post("/estimate", response_model=CausalEstimateResponse)
async def estimate_causal_effect(request: CausalEstimateRequest):
    """Estimate Average Treatment Effect using Double Machine Learning."""
    try:
        res = CausalAnalysisService.estimate_treatment_effect(
            records=request.records,
            treatment_column=request.treatment_column,
            outcome_column=request.outcome_column,
            confounders=request.confounders,
            method=request.method,
        )
        return res
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Causal estimation failed: {str(e)}",
        )
