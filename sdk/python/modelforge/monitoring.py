"""
ModelForge AI SDK - Monitoring, Drift & Pipelines
"""

from typing import Any, Dict, List, Optional


class MonitoringManager:
    def __init__(self, client):
        self.client = client

    def get_metrics(self, deployment_id: str) -> List[Dict[str, Any]]:
        """Fetch real-time throughput and latency percentiles."""
        res = self.client.request("GET", f"/monitoring/{deployment_id}/metrics")
        return res.get("data", [])

    def analyze_drift(
        self,
        deployment_id: str,
        baseline_dataset_version_id: str,
        features: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compute PSI, KS-test, and Wasserstein distance drift metrics."""
        payload = {
            "deployment_id": deployment_id,
            "baseline_dataset_version_id": baseline_dataset_version_id,
            "features_to_monitor": features or [],
        }
        res = self.client.request("POST", "/drift/analyze", json_data=payload)
        return res.get("data", {})


class PipelineManager:
    def __init__(self, client):
        self.client = client

    def list(self, project_id: str) -> List[Dict[str, Any]]:
        res = self.client.request("GET", f"/pipelines?project_id={project_id}")
        return res.get("data", [])

    def run(self, pipeline_id: str) -> Dict[str, Any]:
        res = self.client.request("POST", f"/pipelines/{pipeline_id}/run")
        return res.get("data", {})
