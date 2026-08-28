"""
ModelForge AI - Monitoring & Drift Detection Domain Services
Aggregates application throughput, latency percentiles, error rates, and executes statistical drift analysis.
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.exceptions import EntityNotFoundException
from app.models.monitoring import ModelMonitoringMetric, SystemResourceSnapshot
from app.models.drift import DriftEvent
from app.models.prediction import PredictionLog
from app.models.deployment import Deployment
from app.models.dataset import DatasetVersion
from app.schemas.drift import DriftAnalysisRequest, DriftReportResponse
from app.services.dataset_service import DatasetService
from ml_engine.drift.drift_detector import DriftDetector


class MonitoringService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_deployment_metrics(
        self,
        deployment_id: str,
        minutes_ago: int = 60,
    ) -> List[ModelMonitoringMetric]:
        """Fetch historical monitoring timeseries for deployment."""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
        query = (
            select(ModelMonitoringMetric)
            .where(ModelMonitoringMetric.deployment_id == deployment_id, ModelMonitoringMetric.timestamp >= cutoff)
            .order_by(ModelMonitoringMetric.timestamp.asc())
        )
        res = await self.session.execute(query)
        return list(res.scalars().all())

    async def compute_current_window_metrics(self, deployment_id: str) -> ModelMonitoringMetric:
        """Aggregate last 5 minutes of prediction logs into a monitoring snapshot."""
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(minutes=5)

        query = select(PredictionLog).where(
            PredictionLog.deployment_id == deployment_id,
            PredictionLog.created_at >= window_start,
        )
        res = await self.session.execute(query)
        logs = list(res.scalars().all())

        if not logs:
            metric = ModelMonitoringMetric(
                deployment_id=deployment_id,
                timestamp=now,
                request_count=0,
                error_count=0,
                error_rate=0.0,
                latency_p50=0.0,
                latency_p95=0.0,
                latency_p99=0.0,
            )
            self.session.add(metric)
            await self.session.commit()
            return metric

        latencies = [l.latency_ms for l in logs]
        errors = [l for l in logs if l.status_code >= 400]

        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        p99 = float(np.percentile(latencies, 99))
        error_rate = float(len(errors) / len(logs))

        # Check for ground truth feedback
        feedback_logs = [l for l in logs if l.ground_truth is not None]
        observed_acc = None
        if feedback_logs:
            matches = sum(
                1 for l in feedback_logs
                if l.prediction.get("result") == l.ground_truth.get("actual")
            )
            observed_acc = float(matches / len(feedback_logs))

        metric = ModelMonitoringMetric(
            deployment_id=deployment_id,
            timestamp=now,
            request_count=len(logs),
            error_count=len(errors),
            error_rate=round(error_rate, 4),
            latency_p50=round(p50, 2),
            latency_p95=round(p95, 2),
            latency_p99=round(p99, 2),
            observed_accuracy=round(observed_acc, 4) if observed_acc is not None else None,
        )
        self.session.add(metric)
        await self.session.commit()
        return metric


class DriftService:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.dataset_service = DatasetService(session)

    async def analyze_drift(self, payload: DriftAnalysisRequest) -> DriftEvent:
        """Run statistical drift analysis between baseline training dataset and live inferences / target dataset."""
        query = select(Deployment).where(Deployment.id == payload.deployment_id)
        res = await self.session.execute(query)
        deployment = res.scalar_one_or_none()
        if not deployment:
            raise EntityNotFoundException("Deployment", payload.deployment_id)

        # 1. Fetch Target Data (either from dataset version or latest prediction logs)
        if payload.target_dataset_version_id:
            target_df = await self.dataset_service.get_dataframe_for_version(payload.target_dataset_version_id)
        else:
            # Fetch latest prediction logs
            log_query = select(PredictionLog).where(PredictionLog.deployment_id == deployment.id).order_by(PredictionLog.created_at.desc()).limit(payload.sample_size)
            log_res = await self.session.execute(log_query)
            logs = list(log_res.scalars().all())
            if not logs:
                # Mock minimal dataframe for demonstration
                target_df = pd.DataFrame()
            else:
                features_list = [l.features for l in logs]
                target_df = pd.DataFrame(features_list)

        # 2. Fetch Baseline Data
        if payload.baseline_dataset_version_id:
            baseline_df = await self.dataset_service.get_dataframe_for_version(payload.baseline_dataset_version_id)
        else:
            # Sample baseline from model training dataset
            query = select(DatasetVersion)
            res = await self.session.execute(query)
            first_ver = res.scalars().first()
            if first_ver:
                baseline_df = await self.dataset_service.get_dataframe_for_version(first_ver.id)
            else:
                baseline_df = target_df.copy()

        # 3. Evaluate Drift via DriftDetector
        drift_results = DriftDetector.evaluate_feature_drift(
            baseline_df=baseline_df,
            target_df=target_df,
            psi_threshold_warning=payload.psi_threshold_warning,
            psi_threshold_critical=payload.psi_threshold_critical,
        )

        event = DriftEvent(
            deployment_id=payload.deployment_id,
            drift_type="data_drift",
            severity=drift_results["severity"],
            overall_drift_score=drift_results["overall_drift_score"],
            drifted_features_count=drift_results["drifted_features_count"],
            total_features_count=drift_results["total_features_count"],
            baseline_dataset_version_id=payload.baseline_dataset_version_id,
            target_sample_size=len(target_df),
            feature_metrics=drift_results["feature_metrics"],
            recommendations=drift_results["recommendations"],
            is_resolved=False,
            retraining_triggered=False,
        )
        self.session.add(event)
        await self.session.commit()
        return event
