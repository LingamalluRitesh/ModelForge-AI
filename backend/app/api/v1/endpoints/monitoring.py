"""
ModelForge AI - Monitoring, Drift, Retraining, Explainability, Pipeline & Audit API Endpoints
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.deps import get_db, get_current_user
from app.models.user import User
from app.models.monitoring import ModelMonitoringMetric, SystemResourceSnapshot
from app.models.drift import DriftEvent
from app.models.retraining import RetrainingPolicy, RetrainingExecution
from app.models.explainability import SHAPExplanation, FairnessAnalysisReport
from app.models.pipeline import MLPipeline, PipelineRun
from app.models.alert import Alert, AuditLog
from app.schemas.monitoring import ModelMonitoringMetricResponse, SystemResourceResponse
from app.schemas.drift import DriftAnalysisRequest, DriftReportResponse
from app.schemas.retraining import RetrainingPolicyCreate, RetrainingPolicyResponse, RetrainingExecutionResponse
from app.schemas.explainability import SHAPExplanationRequest, SHAPExplanationResponse, FairnessAnalysisRequest, FairnessReportResponse
from app.schemas.pipeline import MLPipelineCreate, MLPipelineResponse, PipelineRunResponse
from app.schemas.alert import AlertResponse, AuditLogResponse
from app.schemas.common import APIResponse
from app.services.monitoring_service import MonitoringService, DriftService
from app.services.retraining_service import (
    RetrainingService, ExplainabilityService, PipelineService, AlertService, AuditService
)

mon_router = APIRouter(prefix="/monitoring", tags=["Model Monitoring & Observability"])
drift_router = APIRouter(prefix="/drift", tags=["Drift Detection Engine"])
retrain_router = APIRouter(prefix="/retraining", tags=["Automated Retraining"])
xai_router = APIRouter(prefix="/explainability", tags=["Explainable AI & Fairness"])
pipe_router = APIRouter(prefix="/pipelines", tags=["Visual DAG Pipelines"])
alert_router = APIRouter(prefix="/alerts", tags=["Alerting & Notifications"])
audit_router = APIRouter(prefix="/audit", tags=["Immutable Audit Trail"])


# Monitoring
@mon_router.get("/{deployment_id}/metrics", response_model=APIResponse[List[ModelMonitoringMetricResponse]])
async def get_metrics(
    deployment_id: str,
    minutes_ago: int = 60,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    mon_svc = MonitoringService(db)
    metrics = await mon_svc.get_deployment_metrics(deployment_id, minutes_ago)
    return APIResponse(data=[ModelMonitoringMetricResponse.model_validate(m) for m in metrics])


# Drift
@drift_router.post("/analyze", response_model=APIResponse[DriftReportResponse])
async def analyze_drift(
    payload: DriftAnalysisRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    drift_svc = DriftService(db)
    event = await drift_svc.analyze_drift(payload)
    return APIResponse(data=DriftReportResponse.model_validate(event), message="Drift analysis completed.")


# Retraining
@retrain_router.get("/policies", response_model=APIResponse[List[RetrainingPolicyResponse]])
async def list_retraining_policies(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    retrain_svc = RetrainingService(db)
    policies = await retrain_svc.list_policies(project_id)
    return APIResponse(data=[RetrainingPolicyResponse.model_validate(p) for p in policies])


@retrain_router.post("/policies", response_model=APIResponse[RetrainingPolicyResponse], status_code=status.HTTP_201_CREATED)
async def create_retraining_policy(
    project_id: str,
    payload: RetrainingPolicyCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    retrain_svc = RetrainingService(db)
    policy = await retrain_svc.create_policy(project_id, payload)
    return APIResponse(data=RetrainingPolicyResponse.model_validate(policy), message="Retraining policy created.")


@retrain_router.post("/policies/{policy_id}/trigger", response_model=APIResponse[RetrainingExecutionResponse])
async def trigger_retraining(
    policy_id: str,
    reason: str = "Manual trigger",
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    retrain_svc = RetrainingService(db)
    execution = await retrain_svc.execute_retraining_trigger(policy_id, reason)
    return APIResponse(data=RetrainingExecutionResponse.model_validate(execution), message="Retraining execution completed.")


# Explainability
@xai_router.post("/shap/global", response_model=APIResponse[SHAPExplanationResponse])
async def compute_global_shap(
    payload: SHAPExplanationRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    xai_svc = ExplainabilityService(db)
    exp = await xai_svc.compute_global_shap(payload)
    return APIResponse(data=SHAPExplanationResponse.model_validate(exp))


@xai_router.post("/fairness", response_model=APIResponse[FairnessReportResponse])
async def audit_fairness(
    payload: FairnessAnalysisRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    xai_svc = ExplainabilityService(db)
    report = await xai_svc.audit_fairness(payload)
    return APIResponse(data=FairnessReportResponse.model_validate(report))


# Pipelines
@pipe_router.get("", response_model=APIResponse[List[MLPipelineResponse]])
async def list_pipelines(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    pipe_svc = PipelineService(db)
    pipelines = await pipe_svc.list_pipelines(project_id)
    return APIResponse(data=[MLPipelineResponse.model_validate(p) for p in pipelines])


@pipe_router.post("", response_model=APIResponse[MLPipelineResponse], status_code=status.HTTP_201_CREATED)
async def create_pipeline(
    project_id: str,
    payload: MLPipelineCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    pipe_svc = PipelineService(db)
    pipeline = await pipe_svc.create_pipeline(project_id, current_user.id, payload)
    return APIResponse(data=MLPipelineResponse.model_validate(pipeline), message="Pipeline created successfully.")


@pipe_router.post("/{pipeline_id}/run", response_model=APIResponse[PipelineRunResponse])
async def run_pipeline(
    pipeline_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    pipe_svc = PipelineService(db)
    run = await pipe_svc.trigger_pipeline_run(pipeline_id)
    return APIResponse(data=PipelineRunResponse.model_validate(run), message="Pipeline executed successfully.")


# Alerts
@alert_router.get("", response_model=APIResponse[List[AlertResponse]])
async def list_alerts(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    alert_svc = AlertService(db)
    alerts = await alert_svc.list_alerts(project_id)
    return APIResponse(data=[AlertResponse.model_validate(a) for a in alerts])


@alert_router.post("/{alert_id}/acknowledge", response_model=APIResponse[AlertResponse])
async def acknowledge_alert(
    alert_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    alert_svc = AlertService(db)
    alert = await alert_svc.acknowledge_alert(alert_id, current_user.id)
    return APIResponse(data=AlertResponse.model_validate(alert), message="Alert acknowledged.")


# Audit
@audit_router.get("", response_model=APIResponse[List[AuditLogResponse]])
async def list_audit_logs(
    project_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    query = select(AuditLog).order_by(AuditLog.timestamp.desc()).limit(100)
    if project_id:
        query = query.where(AuditLog.project_id == project_id)
    res = await db.execute(query)
    logs = list(res.scalars().all())
    return APIResponse(data=[AuditLogResponse.model_validate(l) for l in logs])
