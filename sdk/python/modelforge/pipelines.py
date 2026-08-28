"""
ModelForge AI Python SDK: Pipeline DAG Orchestration Client
Client interfaces for defining, scheduling, and triggering end-to-end Machine Learning workflows.
"""

from typing import Any, Callable, Dict, List, Optional
import requests


class PipelineClient:
    """Client for triggering and monitoring automated ML DAG pipelines."""

    def __init__(self, client_or_base_url: Any = "http://localhost:8000/api/v1", api_key: Optional[str] = None):
        if hasattr(client_or_base_url, "base_url"):
            self.client = client_or_base_url
            self.base_url = str(client_or_base_url.base_url).rstrip("/")
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {getattr(client_or_base_url, 'api_key', '')}" if getattr(client_or_base_url, 'api_key', None) else "",
            }
        else:
            self.client = None
            self.base_url = str(client_or_base_url).rstrip("/")
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}" if api_key else "",
            }

    def trigger_pipeline(self, pipeline_id: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Trigger immediate pipeline run."""
        payload = {"context": context or {}}
        resp = requests.post(f"{self.base_url}/pipelines/{pipeline_id}/run", json=payload, headers=self.headers)
        if resp.status_code != 200:
            raise RuntimeError(f"Pipeline execution trigger failed: {resp.text}")
        return resp.json()


PipelineManager = PipelineClient
