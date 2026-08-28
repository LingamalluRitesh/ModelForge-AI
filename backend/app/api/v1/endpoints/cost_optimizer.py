"""
ModelForge AI - Cost Optimizer API Endpoints
"""

from typing import Any, Dict
from fastapi import APIRouter, HTTPException, status
from app.services.cost_monitoring_service import CostMonitoringService

router = APIRouter()


@router.get("/projects/{project_id}/summary")
async def get_cost_summary(project_id: str):
    """Retrieve real-time compute cost breakdown and quantization infrastructure savings."""
    try:
        res = CostMonitoringService.get_project_cost_summary(project_id)
        return res
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cost summary fetch failed: {str(e)}",
        )
