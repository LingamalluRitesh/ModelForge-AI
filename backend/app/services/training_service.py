"""
ModelForge AI - Experiment & Training Domain Services
Handles Experiment Tracking, Hyperparameter Logging, Model Training Execution, and Model Comparisons.
"""

import pickle
import time
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sklearn.model_selection import train_test_split

from app.core.exceptions import EntityNotFoundException, MLModelExecutionException
from app.core.storage import storage_engine
from app.models.experiment import Experiment, ExperimentRun
from app.models.training import TrainingJob
from app.schemas.experiment import ExperimentCreate, ExperimentRunCreate
from app.schemas.training import TrainingJobCreate
from app.services.dataset_service import DatasetService
from ml_engine.preprocessing.imputers import AdvancedImputer, AdvancedScaler
from ml_engine.preprocessing.encoders import AdvancedCategoricalEncoder
from ml_engine.algorithms.classification import get_classifier
from ml_engine.algorithms.regression import get_regressor
from ml_engine.evaluation.classification_evaluator import ClassificationEvaluator, RegressionEvaluator


class ExperimentService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def list_experiments(self, project_id: str) -> List[Experiment]:
        query = select(Experiment).where(Experiment.project_id == project_id, Experiment.is_archived == False)
        res = await self.session.execute(query)
        return list(res.scalars().all())

    async def create_experiment(self, project_id: str, user_id: str, payload: ExperimentCreate) -> Experiment:
        exp = Experiment(
            project_id=project_id,
            name=payload.name,
            description=payload.description,
            tags=payload.tags or [],
            created_by_id=user_id,
        )
        self.session.add(exp)
        await self.session.commit()
        return exp

    async def get_experiment_by_id(self, exp_id: str) -> Experiment:
        query = select(Experiment).where(Experiment.id == exp_id)
        res = await self.session.execute(query)
        exp = res.scalar_one_or_none()
        if not exp:
            raise EntityNotFoundException("Experiment", exp_id)
        return exp

    async def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare metrics and hyperparameters of multiple runs side-by-side."""
        query = select(ExperimentRun).where(ExperimentRun.id.in_(run_ids))
        res = await self.session.execute(query)
        runs = list(res.scalars().all())

        metric_keys = set()
        param_keys = set()
        for r in runs:
            if r.metrics:
                metric_keys.update(r.metrics.keys())
            if r.hyperparameters:
                param_keys.update(r.hyperparameters.keys())

        # Determine best run per metric
        best_per_metric = {}
        for m_key in metric_keys:
            valid_runs = [r for r in runs if r.metrics and m_key in r.metrics and isinstance(r.metrics[m_key], (int, float))]
            if valid_runs:
                # Maximize standard metrics
                best_run = max(valid_runs, key=lambda x: x.metrics[m_key])
                best_per_metric[m_key] = best_run.id

        return {
            "runs": runs,
            "metric_keys": sorted(list(metric_keys)),
            "hyperparameter_keys": sorted(list(param_keys)),
            "best_run_id_per_metric": best_per_metric,
        }


class TrainingService:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.dataset_service = DatasetService(session)

    async def execute_training_job(self, project_id: str, user_id: str, payload: TrainingJobCreate) -> Tuple[TrainingJob, ExperimentRun]:
        """Train classifier or regressor model synchronously/asynchronously, evaluate, and save artifact."""
        start_time = time.time()

        # Load Dataset
        df = await self.dataset_service.get_dataframe_for_version(payload.dataset_version_id)
        if payload.target_column not in df.columns:
            raise MLModelExecutionException(f"Target column '{payload.target_column}' not found in dataset.")

        # Create Training Job record
        job = TrainingJob(
            project_id=project_id,
            experiment_id=payload.experiment_id,
            name=payload.name,
            algorithm=payload.algorithm,
            dataset_version_id=payload.dataset_version_id,
            target_column=payload.target_column,
            feature_columns=payload.feature_columns,
            hyperparameters=payload.hyperparameters or {},
            status="running",
            progress_percentage=20.0,
            created_by_id=user_id,
            started_at=datetime.now(timezone.utc),
        )
        self.session.add(job)
        await self.session.flush()

        # Preprocessing
        selected_cols = [c for c in payload.feature_columns if c in df.columns]
        X_df = df[selected_cols].copy()
        y_series = df[payload.target_column].copy()

        imputer = AdvancedImputer()
        encoder = AdvancedCategoricalEncoder()
        scaler = AdvancedScaler()

        X_imp = imputer.fit_transform(X_df)
        X_enc = encoder.fit_transform(X_imp)
        X_scaled = scaler.fit_transform(X_enc)

        X_arr = X_scaled.values
        y_arr = y_series.values

        is_cls = not pd.api.types.is_float_dtype(y_series) and y_series.nunique() <= 50

        X_train, X_test, y_train, y_test = train_test_split(
            X_arr, y_arr, test_size=payload.validation_split, random_state=payload.random_seed,
            stratify=y_arr if is_cls and len(np.unique(y_arr)) > 1 else None,
        )

        job.progress_percentage = 60.0
        await self.session.flush()

        # Train model
        params = payload.hyperparameters or {}
        if is_cls:
            model = get_classifier(payload.algorithm, **params)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)
            metrics = ClassificationEvaluator.evaluate(y_test, preds, probs)
        else:
            model = get_regressor(payload.algorithm, **params)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics = RegressionEvaluator.evaluate(y_test, preds)

        duration = time.time() - start_time

        # Bundle model & preprocessing pipeline into deployable artifact
        pipeline_artifact = {
            "model": model,
            "imputer": imputer,
            "encoder": encoder,
            "scaler": scaler,
            "feature_columns": selected_cols,
            "target_column": payload.target_column,
            "problem_type": "classification" if is_cls else "regression",
            "metrics": metrics,
        }

        artifact_bytes = pickle.dumps(pipeline_artifact)
        artifact_key = f"projects/{project_id}/models/{job.id}/model_artifact.pkl"
        storage_uri = await storage_engine.save_file(artifact_bytes, artifact_key)

        # Create Experiment Run record
        run = ExperimentRun(
            experiment_id=payload.experiment_id,
            name=f"{payload.name}-run",
            dataset_version_id=payload.dataset_version_id,
            algorithm_name=payload.algorithm,
            framework="scikit-learn" if "pytorch" not in payload.algorithm else "pytorch",
            status="completed",
            duration_seconds=round(duration, 2),
            hyperparameters=params,
            metrics=metrics,
            model_artifact_uri=storage_uri,
            created_by_id=user_id,
            completed_at=datetime.now(timezone.utc),
        )
        self.session.add(run)
        await self.session.flush()

        # Update Job status
        job.status = "completed"
        job.progress_percentage = 100.0
        job.metrics = metrics
        job.run_id = run.id
        job.completed_at = datetime.now(timezone.utc)

        await self.session.commit()
        return job, run
