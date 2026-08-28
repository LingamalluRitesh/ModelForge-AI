"""
ModelForge AI SDK - Pipeline Manager
"""

from typing import Any, Dict, List, Optional


class PipelineManager:
    """Manages DAG execution pipelines."""

    def __init__(self, client):
        self.client = client

    def list(self, project_id: str) -> List[Any]:
        resp = self.client.get(f"/pipelines?project_id={project_id}")
        return resp.get("data", [])

    def trigger(self, pipeline_id: str) -> Dict[str, Any]:
        resp = self.client.post(f"/pipelines/{pipeline_id}/runs", json={})
        return resp.get("data", {})
