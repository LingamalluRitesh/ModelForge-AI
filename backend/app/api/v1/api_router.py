"""
ModelForge AI - API v1 Master Router
"""

from fastapi import APIRouter
from app.api.v1.endpoints.auth import router as auth_router
from app.api.v1.endpoints.projects import router as projects_router, org_router
from app.api.v1.endpoints.datasets import router as datasets_router
from app.api.v1.endpoints.experiments import (
    router as experiments_router,
    train_router,
    automl_router,
)
from app.api.v1.endpoints.registry import (
    reg_router,
    deploy_router,
    pred_router,
)
from app.api.v1.endpoints.monitoring import (
    mon_router,
    drift_router,
    retrain_router,
    xai_router,
    pipe_router,
    alert_router,
    audit_router,
)

api_router = APIRouter()

api_router.include_router(auth_router)
api_router.include_router(org_router)
api_router.include_router(projects_router)
api_router.include_router(datasets_router)
api_router.include_router(experiments_router)
api_router.include_router(train_router)
api_router.include_router(automl_router)
api_router.include_router(reg_router)
api_router.include_router(deploy_router)
api_router.include_router(pred_router)
api_router.include_router(mon_router)
api_router.include_router(drift_router)
api_router.include_router(retrain_router)
api_router.include_router(xai_router)
api_router.include_router(pipe_router)
api_router.include_router(alert_router)
api_router.include_router(audit_router)
