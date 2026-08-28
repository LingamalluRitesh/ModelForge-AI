"""
ModelForge AI - Deployment & Real-Time Inference Domain Services
Handles Real-Time REST Serving, Canary Stage Transitions (5% -> 20% -> 50% -> 100%),
A/B Traffic Splitting, Automated Rollbacks, and Batch Inference Jobs.
"""

import pickle
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.exceptions import EntityNotFoundException, ValidationException, MLModelExecutionException
from app.core.storage import storage_engine
from app.models.deployment import Deployment, DeploymentRollbackHistory
from app.models.model_registry import ModelVersion
from app.models.prediction import PredictionLog, BatchInferenceJob
from app.models.monitoring import ModelMonitoringMetric
from app.schemas.deployment import DeploymentCreate, DeploymentUpdate, CanaryUpdateRequest, ABTestTrafficUpdateRequest, RollbackRequest
from app.schemas.prediction import RealtimePredictionRequest, RealtimePredictionResponse


# Global in-memory model cache for microsecond inference routing
_loaded_model_cache: Dict[str, Any] = {}


class DeploymentService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def list_deployments(self, project_id: str) -> List[Deployment]:
        query = select(Deployment).where(Deployment.project_id == project_id)
        res = await self.session.execute(query)
        return list(res.scalars().all())

    async def get_deployment_by_id(self, deployment_id: str) -> Deployment:
        query = select(Deployment).where(Deployment.id == deployment_id)
        res = await self.session.execute(query)
        dep = res.scalar_one_or_none()
        if not dep:
            raise EntityNotFoundException("Deployment", deployment_id)
        return dep

    async def create_deployment(self, project_id: str, user_id: str, payload: DeploymentCreate) -> Deployment:
        # Check endpoint uniqueness
        query = select(Deployment).where(Deployment.endpoint_path == payload.endpoint_path)
        res = await self.session.execute(query)
        if res.scalar_one_or_none():
            raise ValidationException(f"Endpoint path '{payload.endpoint_path}' is already in use.")

        deployment = Deployment(
            project_id=project_id,
            name=payload.name,
            endpoint_path=payload.endpoint_path,
            environment=payload.environment,
            model_version_id=payload.model_version_id,
            strategy=payload.strategy,
            secondary_model_version_id=payload.secondary_model_version_id,
            primary_traffic_percentage=payload.primary_traffic_percentage,
            min_replicas=payload.min_replicas,
            max_replicas=payload.max_replicas,
            cpu_limit=payload.cpu_limit,
            memory_limit=payload.memory_limit,
            error_rate_threshold=payload.error_rate_threshold,
            latency_threshold_ms=payload.latency_threshold_ms,
            auto_rollback_enabled=payload.auto_rollback_enabled,
            created_by_id=user_id,
        )
        self.session.add(deployment)
        await self.session.commit()
        return deployment

    async def update_canary_stage(self, deployment_id: str, payload: CanaryUpdateRequest) -> Deployment:
        """Advance Canary rollout stage (e.g. 5% -> 20% -> 50% -> 100%)."""
        deployment = await self.get_deployment_by_id(deployment_id)
        deployment.canary_stage_percentage = payload.canary_stage_percentage

        if payload.canary_stage_percentage >= 100.0 and deployment.secondary_model_version_id:
            # Promote canary model to primary model
            deployment.model_version_id = deployment.secondary_model_version_id
            deployment.secondary_model_version_id = None
            deployment.strategy = "direct"
            deployment.primary_traffic_percentage = 100.0
            deployment.canary_stage_percentage = 0.0

        await self.session.commit()
        return deployment

    async def update_ab_traffic(self, deployment_id: str, payload: ABTestTrafficUpdateRequest) -> Deployment:
        deployment = await self.get_deployment_by_id(deployment_id)
        deployment.strategy = "ab_test"
        deployment.secondary_model_version_id = payload.secondary_model_version_id
        deployment.primary_traffic_percentage = payload.primary_traffic_percentage
        await self.session.commit()
        return deployment

    async def execute_rollback(self, deployment_id: str, payload: RollbackRequest, trigger: str = "manual") -> Deployment:
        deployment = await self.get_deployment_by_id(deployment_id)
        old_version_id = deployment.model_version_id

        # Update deployment to target model version
        deployment.model_version_id = payload.target_model_version_id
        deployment.strategy = "direct"
        deployment.secondary_model_version_id = None
        deployment.primary_traffic_percentage = 100.0

        # Record rollback in audit history
        history = DeploymentRollbackHistory(
            deployment_id=deployment_id,
            from_model_version_id=old_version_id,
            to_model_version_id=payload.target_model_version_id,
            trigger_type=trigger,
            reason=payload.reason,
        )
        self.session.add(history)
        await self.session.commit()
        return deployment


class InferenceService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def _load_model_artifact(self, version_id: str) -> Dict[str, Any]:
        """Fetch and cache unpickled pipeline artifact in memory."""
        global _loaded_model_cache
        if version_id in _loaded_model_cache:
            return _loaded_model_cache[version_id]

        query = select(ModelVersion).where(ModelVersion.id == version_id)
        res = await self.session.execute(query)
        version = res.scalar_one_or_none()
        if not version:
            raise EntityNotFoundException("ModelVersion", version_id)

        artifact_bytes = await storage_engine.read_file(version.storage_uri)
        pipeline_obj = pickle.loads(artifact_bytes)
        _loaded_model_cache[version_id] = pipeline_obj
        return pipeline_obj

    async def predict_realtime(
        self,
        endpoint_path: str,
        payload: RealtimePredictionRequest,
    ) -> RealtimePredictionResponse:
        """Execute real-time prediction with canary/AB traffic routing and latency tracking."""
        start_time = time.time()
        request_id = str(uuid.uuid4())

        query = select(Deployment).where(Deployment.endpoint_path == endpoint_path, Deployment.status == "active")
        res = await self.session.execute(query)
        deployment = res.scalar_one_or_none()
        if not deployment:
            raise EntityNotFoundException("DeploymentEndpoint", endpoint_path)

        # Traffic Routing
        selected_version_id = deployment.model_version_id
        if deployment.strategy == "canary" and deployment.secondary_model_version_id:
            rand_val = np.random.uniform(0, 100)
            if rand_val < deployment.canary_stage_percentage:
                selected_version_id = deployment.secondary_model_version_id
        elif deployment.strategy == "ab_test" and deployment.secondary_model_version_id:
            rand_val = np.random.uniform(0, 100)
            if rand_val > deployment.primary_traffic_percentage:
                selected_version_id = deployment.secondary_model_version_id

        # Load pipeline artifact
        pipeline = await self._load_model_artifact(selected_version_id)
        model = pipeline["model"]
        imputer = pipeline["imputer"]
        encoder = pipeline["encoder"]
        scaler = pipeline["scaler"]

        # Transform single input instance
        input_df = pd.DataFrame([payload.features])
        X_imp = imputer.transform(input_df)
        X_enc = encoder.transform(X_imp)
        X_scaled = scaler.transform(X_enc)

        # Inference
        preds = model.predict(X_scaled.values)
        prediction_val = preds[0]

        prob = None
        prob_dict = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_scaled.values)[0]
            prob = float(np.max(probs))
            if hasattr(model, "classes_") and model.classes_ is not None:
                prob_dict = {str(c): float(p) for c, p in zip(model.classes_, probs)}

        latency_ms = (time.time() - start_time) * 1000.0

        # Log prediction to database
        pred_log = PredictionLog(
            deployment_id=deployment.id,
            model_version_id=selected_version_id,
            request_id=request_id,
            features=payload.features,
            prediction={"result": int(prediction_val) if isinstance(prediction_val, (np.integer, bool)) else (float(prediction_val) if isinstance(prediction_val, np.floating) else str(prediction_val))},
            probability_or_confidence=prob,
            latency_ms=round(latency_ms, 2),
            status_code=200,
        )
        self.session.add(pred_log)
        await self.session.commit()

        return RealtimePredictionResponse(
            prediction=int(prediction_val) if isinstance(prediction_val, (np.integer, bool)) else (float(prediction_val) if isinstance(prediction_val, np.floating) else str(prediction_val)),
            probability_or_confidence=round(prob, 4) if prob is not None else None,
            probabilities=prob_dict,
            model_name=deployment.name,
            model_version_tag=selected_version_id,
            deployment_id=deployment.id,
            request_id=request_id,
            latency_ms=round(latency_ms, 2),
            timestamp=datetime.now(timezone.utc),
        )
