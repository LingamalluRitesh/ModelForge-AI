"""
ModelForge AI - Federated Learning API Endpoints
"""

from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from app.services.federated_coordinator_service import FederatedCoordinatorService

router = APIRouter()


class ClientUpdateSubmission(BaseModel):
    session_id: str
    client_id: str
    round_number: int
    weights: List[List[float]]
    num_samples: int
    loss: float


class AggregateRequest(BaseModel):
    client_updates: List[ClientUpdateSubmission]
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5


@router.post("/aggregate")
async def aggregate_federated_round(request: AggregateRequest):
    """Aggregate decentralized edge client weights using FedAvg with DP noise."""
    try:
        res = FederatedCoordinatorService.aggregate_round_updates(
            client_updates=[u.dict() for u in request.client_updates],
            dp_epsilon=request.dp_epsilon,
            dp_delta=request.dp_delta,
        )
        return res
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Federated aggregation failed: {str(e)}",
        )
