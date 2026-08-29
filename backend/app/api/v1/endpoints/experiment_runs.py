"""
Experiment Runs REST Endpoints.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

from app.services.experiment_run_exporter import ExperimentRunExporter

router = APIRouter(prefix="/experiment-runs", tags=["Experiment Runs Exporter"])
exporter = ExperimentRunExporter()


class CreateRunRequest(BaseModel):
    experiment_id: str
    run_name: str
    parameters: Optional[Dict[str, Any]] = None
    tags: Optional[Dict[str, str]] = None


class LogMetricRequest(BaseModel):
    key: str
    value: float
    step: int


@router.post("/create", summary="Create New Experiment Run")
def create_run(req: CreateRunRequest) -> Dict[str, Any]:
    return exporter.create_run(req.experiment_id, req.run_name, req.parameters, req.tags)


@router.post("/{run_id}/metrics", summary="Log Metric Step")
def log_metric(run_id: str, req: LogMetricRequest) -> Dict[str, Any]:
    try:
        exporter.log_metric(run_id, req.key, req.value, req.step)
        return {"status": "METRIC_LOGGED", "run_id": run_id, "key": req.key}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{run_id}/export/mlflow", summary="Export in MLflow Format")
def export_mlflow(run_id: str) -> Dict[str, Any]:
    try:
        return exporter.export_mlflow_format(run_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
