"""
ModelForge AI - Cloud Cost & Compute Optimization Service
Analyzes inference workloads, calculates cost per million predictions, and recommends instance right-sizing.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timezone, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.models.deployment import Deployment
from app.models.monitoring import ModelMonitoringMetric


class CostOptimizerService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_deployment_cost_analysis(self, deployment_id: str) -> Dict[str, Any]:
        res = await self.session.execute(select(Deployment).where(Deployment.id == deployment_id))
        dep = res.scalar_one_or_none()
        if not dep:
            return {"error": "Deployment not found"}

        # Approximate cost calculation based on replica count and compute tier
        hourly_rate_per_replica = 0.096 # e.g. AWS c6i.large or GCP e2-standard-2
        replicas = dep.current_replicas
        monthly_cost = hourly_rate_per_replica * replicas * 730.0

        # Query recent request count
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        query = (
            select(func.sum(ModelMonitoringMetric.request_count))
            .where(ModelMonitoringMetric.deployment_id == deployment_id)
            .where(ModelMonitoringMetric.timestamp >= cutoff)
        )
        count_res = await self.session.execute(query)
        total_requests_30d = count_res.scalar() or 100000

        cost_per_million = (monthly_cost / max(1, total_requests_30d)) * 1_000_000

        recommendation = "Optimal"
        if replicas > 3 and total_requests_30d < 50000:
            recommendation = "Downscale: Reduce min_replicas to 1 to save up to 65% cost."
        elif replicas == dep.max_replicas and total_requests_30d > 5_000_000:
            recommendation = "Upscale: Increase max_replicas or switch to GPU inference tier."

        return {
            "deployment_id": deployment_id,
            "deployment_name": dep.name,
            "current_replicas": replicas,
            "estimated_monthly_cost_usd": round(monthly_cost, 2),
            "total_requests_30d": int(total_requests_30d),
            "cost_per_million_predictions_usd": round(cost_per_million, 4),
            "right_sizing_recommendation": recommendation,
        }
