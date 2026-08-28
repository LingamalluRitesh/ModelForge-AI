"""
ModelForge AI - Celery Asynchronous Worker Tasks
"""

import time
import asyncio
from typing import Any, Dict, List
from workers.celery_app import celery_app
from app.core.database import AsyncSessionLocal
from app.core.logging import logger
from app.services.dataset_service import DatasetService
from app.services.training_service import TrainingService
from app.services.automl_service import AutoMLService
from app.services.monitoring_service import DriftService
from app.services.retraining_service import RetrainingService, PipelineService
from app.schemas.training import TrainingJobCreate, AutoMLJobCreate
from app.schemas.drift import DriftAnalysisRequest


@celery_app.task(name="workers.tasks.process_dataset_task", bind=True)
def process_dataset_task(self, version_id: str):
    """Worker task to run asynchronous data quality analysis and profiling on new datasets."""
    logger.info(f"Worker received process_dataset_task for version '{version_id}'")

    async def _run():
        async with AsyncSessionLocal() as session:
            svc = DatasetService(session)
            await svc.run_data_quality(version_id)

    asyncio.run(_run())
    return {"status": "completed", "version_id": version_id}


@celery_app.task(name="workers.tasks.train_model_task", bind=True)
def train_model_task(self, project_id: str, user_id: str, job_dict: Dict[str, Any]):
    """Worker task to train ML / Deep Learning model asynchronously."""
    logger.info(f"Worker received train_model_task for algorithm '{job_dict.get('algorithm')}'")

    async def _run():
        async with AsyncSessionLocal() as session:
            svc = TrainingService(session)
            payload = TrainingJobCreate(**job_dict)
            job, run = await svc.execute_training_job(project_id, user_id, payload)
            return {"job_id": job.id, "run_id": run.id, "status": "completed"}

    return asyncio.run(_run())


@celery_app.task(name="workers.tasks.run_automl_task", bind=True)
def run_automl_task(self, project_id: str, user_id: str, automl_dict: Dict[str, Any]):
    """Worker task for AutoML pipeline and trial optimization."""
    logger.info(f"Worker received run_automl_task: {automl_dict.get('name')}")

    async def _run():
        async with AsyncSessionLocal() as session:
            svc = AutoMLService(session)
            payload = AutoMLJobCreate(**automl_dict)
            job = await svc.launch_automl_job(project_id, user_id, payload)
            return {"automl_job_id": job.id, "best_algorithm": job.best_algorithm, "best_score": job.best_score}

    return asyncio.run(_run())


@celery_app.task(name="workers.tasks.detect_drift_task", bind=True)
def detect_drift_task(self, drift_dict: Dict[str, Any]):
    """Worker task to run statistical PSI and KS drift checks periodically."""
    logger.info(f"Worker received detect_drift_task for deployment '{drift_dict.get('deployment_id')}'")

    async def _run():
        async with AsyncSessionLocal() as session:
            svc = DriftService(session)
            payload = DriftAnalysisRequest(**drift_dict)
            event = await svc.analyze_drift(payload)
            return {"event_id": event.id, "drift_score": event.overall_drift_score, "severity": event.severity}

    return asyncio.run(_run())


@celery_app.task(name="workers.tasks.trigger_retraining_task", bind=True)
def trigger_retraining_task(self, policy_id: str, reason: str):
    """Worker task to run automated retraining when drift or accuracy drops occur."""
    logger.info(f"Worker received trigger_retraining_task for policy '{policy_id}'")

    async def _run():
        async with AsyncSessionLocal() as session:
            svc = RetrainingService(session)
            execution = await svc.execute_retraining_trigger(policy_id, reason)
            return {"execution_id": execution.id, "status": execution.status}

    return asyncio.run(_run())


@celery_app.task(name="workers.tasks.execute_pipeline_dag_task", bind=True)
def execute_pipeline_dag_task(self, pipeline_id: str):
    """Worker task to execute visual ML DAG pipeline."""
    logger.info(f"Worker executing pipeline DAG '{pipeline_id}'")

    async def _run():
        async with AsyncSessionLocal() as session:
            svc = PipelineService(session)
            run = await svc.trigger_pipeline_run(pipeline_id, trigger_type="celery_worker")
            return {"pipeline_run_id": run.id, "status": run.status}

    return asyncio.run(_run())
