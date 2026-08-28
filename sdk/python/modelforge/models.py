"""
ModelForge AI SDK - Model Registry & Governance
"""

from typing import Any, Dict, List, Optional


class RegisteredModel:
    def __init__(self, data: Dict[str, Any], manager: "ModelRegistryManager"):
        self._data = data
        self._manager = manager

    @property
    def id(self) -> str:
        return self._data.get("id", "")

    @property
    def name(self) -> str:
        return self._data.get("name", "")

    @property
    def versions(self) -> List[Dict[str, Any]]:
        return self._data.get("versions", [])


class ModelRegistryManager:
    def __init__(self, client):
        self.client = client

    def list(self, project_id: str) -> List[RegisteredModel]:
        res = self.client.request("GET", f"/registry/models?project_id={project_id}")
        return [RegisteredModel(m, self) for m in res.get("data", [])]

    def register(
        self,
        project_id: str,
        name: str,
        problem_type: str = "classification",
        description: Optional[str] = None,
    ) -> RegisteredModel:
        payload = {
            "name": name,
            "problem_type": problem_type,
            "description": description,
        }
        res = self.client.request("POST", f"/registry/models?project_id={project_id}", json_data=payload)
        return RegisteredModel(res.get("data", {}), self)

    def create_version(
        self,
        registered_model_id: str,
        version_tag: str,
        experiment_run_id: str,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create model version artifact from an experiment run."""
        payload = {
            "version_tag": version_tag,
            "experiment_run_id": experiment_run_id,
            "description": description,
        }
        res = self.client.request("POST", f"/registry/models/{registered_model_id}/versions", json_data=payload)
        return res.get("data", {})
