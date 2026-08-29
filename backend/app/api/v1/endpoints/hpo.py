"""
Bayesian Hyperparameter Optimization REST Endpoints.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

from app.services.bayesian_hpo_service import BayesianOptimizationEngine

router = APIRouter(prefix="/hpo", tags=["Hyperparameter Optimization"])
engine = BayesianOptimizationEngine()


class TrialReportRequest(BaseModel):
    params: Dict[str, float]
    loss: float
    accuracy: float


@router.get("/suggest", summary="Suggest Next Hyperparameter Trial")
def suggest_hyperparameters() -> Dict[str, Any]:
    params = engine.suggest_parameters()
    return {"suggested_params": params, "total_trials": len(engine.trials)}


@router.post("/report", summary="Report Trial Outcome")
def report_trial(req: TrialReportRequest) -> Dict[str, Any]:
    engine.report_trial(req.params, req.loss, req.accuracy)
    return {"status": "TRIAL_RECORDED", "best": engine.get_best_hyperparameters()}
