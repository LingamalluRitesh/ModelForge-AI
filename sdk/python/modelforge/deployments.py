"""
ModelForge AI SDK - Deployments, Canary Routing & Inferences
"""

from typing import Any, Dict, List, Optional


class Deployment:
    def __init__(self, data: Dict[str, Any], manager: "DeploymentManager"):
        self._data = data
        self._manager = manager

    @property
    def id(self) -> str:
        return self._data.get("id", "")

    @property
    def name(self) -> str:
        return self._data.get("name", "")

    @property
    def endpoint_path(self) -> str:
        return self._data.get("endpoint_path", "")

    @property
    def strategy(self) -> str:
        return self._data.get("strategy", "")

    @property
    def canary_percentage(self) -> float:
        return self._data.get("canary_stage_percentage", 0.0)

    def set_canary_traffic(self, percentage: float) -> Dict[str, Any]:
        """Update Canary rollout percentage (e.g. 5.0 -> 20.0 -> 50.0 -> 100.0)."""
        return self._manager.client.request(
            "POST",
            f"/deployments/{self.id}/canary",
            json_data={"canary_stage_percentage": percentage},
        ).get("data", {})

    def rollback(self, reason: Optional[str] = None) -> Dict[str, Any]:
        """Trigger instant automated rollback to previous healthy version."""
        return self._manager.client.request(
            "POST",
            f"/deployments/{self.id}/rollback",
            json_data={"reason": reason or "Triggered via Python SDK"},
        ).get("data", {})


class DeploymentManager:
    def __init__(self, client):
        self.client = client

    def list(self, project_id: str) -> List[Deployment]:
        res = self.client.request("GET", f"/deployments?project_id={project_id}")
        return [Deployment(d, self) for d in res.get("data", [])]

    def create(
        self,
        project_id: str,
        name: str,
        endpoint_path: str,
        model_version_id: str,
        environment: str = "production",
        strategy: str = "direct",
        min_replicas: int = 2,
    ) -> Deployment:
        payload = {
            "name": name,
            "endpoint_path": endpoint_path,
            "model_version_id": model_version_id,
            "environment": environment,
            "strategy": strategy,
            "min_replicas": min_replicas,
        }
        res = self.client.request("POST", f"/deployments?project_id={project_id}", json_data=payload)
        return Deployment(res.get("data", {}), self)


class PredictionManager:
    def __init__(self, client):
        self.client = client

    def predict(self, endpoint_path: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """Execute real-time low-latency prediction against deployed model."""
        payload = {"features": features}
        return self.client.request("POST", f"/predictions/{endpoint_path}", json_data=payload)

    def predict_batch(self, deployment_id: str, input_file_path: str) -> Dict[str, Any]:
        """Submit asynchronous batch scoring job."""
        files = {"file": open(input_file_path, "rb")}
        return self.client.request("POST", f"/predictions/batch?deployment_id={deployment_id}", files=files)
