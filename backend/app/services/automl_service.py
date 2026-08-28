"""
ModelForge AI - AutoML Domain Service
Orchestrates automated dataset exploration, preprocessing, model candidate generation, and leaderboard creation.
"""

import pickle
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.exceptions import EntityNotFoundException, MLModelExecutionException
from app.core.storage import storage_engine
from app.models.training import AutoMLJob, AutoMLTrial
from app.models.experiment import ExperimentRun
from app.schemas.training import AutoMLJobCreate
from app.services.dataset_service import DatasetService
from ml_engine.automl.automl_engine import AutoMLPipeline


class AutoMLService:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.dataset_service = DatasetService(session)

    async def launch_automl_job(self, project_id: str, user_id: str, payload: AutoMLJobCreate) -> AutoMLJob:
        """Launch complete automated machine learning pipeline."""
        start_time = time.time()

        df = await self.dataset_service.get_dataframe_for_version(payload.dataset_version_id)
        if payload.target_column not in df.columns:
            raise MLModelExecutionException(f"Target column '{payload.target_column}' not found.")

        # Initialize AutoML Job in database
        job = AutoMLJob(
            project_id=project_id,
            experiment_id=payload.experiment_id,
            name=payload.name,
            dataset_version_id=payload.dataset_version_id,
            target_column=payload.target_column,
            problem_type=payload.problem_type or "auto",
            optimization_metric=payload.optimization_metric,
            time_limit_seconds=payload.time_limit_seconds,
            max_trials=payload.max_trials,
            algorithms_to_evaluate=payload.algorithms_to_evaluate or [],
            status="running",
            created_by_id=user_id,
        )
        self.session.add(job)
        await self.session.flush()

        # Run AutoML Pipeline
        automl = AutoMLPipeline(
            target_column=payload.target_column,
            problem_type=payload.problem_type if payload.problem_type != "auto" else None,
            optimization_metric=payload.optimization_metric,
            max_trials_per_algo=max(3, payload.max_trials // 5),
            time_limit_seconds=payload.time_limit_seconds,
        )

        results = automl.run_automl(df)

        job.problem_type = results["problem_type"]
        job.best_algorithm = results["best_algorithm"]
        job.best_score = results["best_score"]
        job.leaderboard = results["leaderboard"]

        # Persist trials and best model run
        for idx, item in enumerate(results["leaderboard"]):
            trial = AutoMLTrial(
                automl_job_id=job.id,
                trial_number=idx + 1,
                algorithm_name=item["algorithm"],
                hyperparameters=item["hyperparameters"],
                metrics=item["metrics"],
                primary_metric_value=item["score"],
                duration_seconds=item["duration_seconds"],
                status="completed",
            )
            self.session.add(trial)

        # Save Best Model Artifact
        if automl.best_model:
            pipeline_artifact = {
                "model": automl.best_model,
                "imputer": automl.imputer,
                "encoder": automl.encoder,
                "scaler": automl.scaler,
                "target_column": payload.target_column,
                "problem_type": results["problem_type"],
                "metrics": results["leaderboard"][0]["metrics"] if results["leaderboard"] else {},
            }
            artifact_bytes = pickle.dumps(pipeline_artifact)
            artifact_key = f"projects/{project_id}/automl/{job.id}/best_model.pkl"
            storage_uri = await storage_engine.save_file(artifact_bytes, artifact_key)

            # Create champion experiment run
            run = ExperimentRun(
                experiment_id=payload.experiment_id,
                name=f"automl-best-{results['best_algorithm']}",
                dataset_version_id=payload.dataset_version_id,
                algorithm_name=results["best_algorithm"] or "unknown",
                framework="scikit-learn",
                status="completed",
                duration_seconds=round(time.time() - start_time, 2),
                hyperparameters=results["leaderboard"][0]["hyperparameters"] if results["leaderboard"] else {},
                metrics=results["leaderboard"][0]["metrics"] if results["leaderboard"] else {},
                model_artifact_uri=storage_uri,
                created_by_id=user_id,
                completed_at=datetime.now(timezone.utc),
            )
            self.session.add(run)
            await self.session.flush()
            job.best_run_id = run.id

        job.status = "completed"
        job.completed_at = datetime.now(timezone.utc)
        await self.session.commit()
        return job
