"""
ModelForge AI Python SDK: Dark Shadow Traffic Client
"""

from typing import Any, Dict, Optional
import requests


class ShadowTrafficClient:
    def __init__(self, base_url: str = "http://localhost:8000/api/v1", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}" if api_key else ""}

    def get_shadow_telemetry(self, deployment_id: str) -> Dict[str, Any]:
        resp = requests.get(f"{self.base_url}/deployments/{deployment_id}/shadow-stats", headers=self.headers)
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to fetch shadow stats: {resp.text}")
        return resp.json()
