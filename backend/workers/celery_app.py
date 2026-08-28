"""
ModelForge AI - Celery Distributed Task Queue Configuration
Routes jobs to dedicated task queues (training, automl, data-processing, drift, retraining).
"""

import os
from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "modelforge_workers",
    broker=settings.celery_broker,
    backend=settings.celery_backend,
    include=["workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=7200,  # 2 hours max for heavy training
    task_routes={
        "workers.tasks.process_dataset_task": {"queue": "data-processing"},
        "workers.tasks.train_model_task": {"queue": "training"},
        "workers.tasks.run_automl_task": {"queue": "automl"},
        "workers.tasks.execute_batch_inference_task": {"queue": "batch-inference"},
        "workers.tasks.detect_drift_task": {"queue": "drift-detection"},
        "workers.tasks.trigger_retraining_task": {"queue": "retraining"},
        "workers.tasks.execute_pipeline_dag_task": {"queue": "pipeline"},
    },
)
