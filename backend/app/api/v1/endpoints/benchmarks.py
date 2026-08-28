"""
ModelForge AI - Benchmarks API Endpoints
"""

from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, status
from app.services.benchmarks_leaderboard_service import BenchmarksLeaderboardService

router = APIRouter()


@router.get("/projects/{project_id}/leaderboard")
async def get_leaderboard(project_id: str):
    """Retrieve model performance rankings and benchmark leaderboard."""
    try:
        res = BenchmarksLeaderboardService.get_project_leaderboard(project_id)
        return res
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch leaderboard: {str(e)}",
        )
