"""
ModelForge AI - Distributed Training API Endpoints
"""

from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from app.services.distributed_training_service import DistributedTrainingService

router = APIRouter()


class LaunchTrainingRequest(BaseModel):
    project_id: str
    name: str
    framework: str = "ray_train"
    num_nodes: int = Field(2, ge=1, le=64)
    gpus_per_node: int = Field(1, ge=0, le=8)
    entrypoint_command: str = "python -m train_distributed"
    hyperparameters: Optional[Dict[str, Any]] = None


@router.post("/launch")
async def launch_distributed_training(request: LaunchTrainingRequest):
    """Launch multi-node distributed training job across Ray / PyTorch DDP workers."""
    try:
        res = DistributedTrainingService.launch_distributed_job(
            project_id=request.project_id,
            name=request.name,
            framework=request.framework,
            num_nodes=request.num_nodes,
            gpus_per_node=request.gpus_per_node,
            entrypoint_command=request.entrypoint_command,
            hyperparameters=request.hyperparameters,
        )
        return res
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to launch distributed training: {str(e)}",
        )
