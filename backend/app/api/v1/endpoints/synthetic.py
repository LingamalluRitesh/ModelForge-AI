"""
ModelForge AI - Synthetic Data API Endpoints
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from app.services.synthetic_data_service import SyntheticDataService

router = APIRouter()


class SyntheticGenerateRequest(BaseModel):
    source_records: List[Dict[str, Any]] = Field(..., description="Source tabular records to model")
    algorithm: str = Field("ctgan", description="ctgan, gaussian_copula, or privbayes")
    num_samples: int = Field(100, ge=1, le=50000)
    continuous_columns: Optional[List[str]] = None


class SyntheticGenerateResponse(BaseModel):
    algorithm: str
    num_samples_generated: int
    fidelity_score: float
    privacy_score: float
    differential_privacy_epsilon: float
    differential_privacy_delta: float
    records: List[Dict[str, Any]]
    summary_statistics: Dict[str, Any]


@router.post("/generate", response_model=SyntheticGenerateResponse)
async def generate_synthetic_data(request: SyntheticGenerateRequest):
    """Generate high-fidelity, privacy-preserving synthetic tabular data."""
    try:
        res = SyntheticDataService.generate_synthetic_batch(
            source_records=request.source_records,
            algorithm=request.algorithm,
            num_samples=request.num_samples,
            continuous_columns=request.continuous_columns,
        )
        return res
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Synthetic data generation failed: {str(e)}",
        )
