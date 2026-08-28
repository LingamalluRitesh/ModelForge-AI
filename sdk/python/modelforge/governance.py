"""
ModelForge AI Python SDK: Governance & Compliance Client
"""

from typing import Any, Dict, Optional
import requests


class GovernanceClient:
    def __init__(self, base_url: str = "http://localhost:8000/api/v1", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}" if api_key else ""}

    def certify_fairness(self, model_version_id: str) -> Dict[str, Any]:
        resp = requests.post(f"{self.base_url}/governance/certify-fairness", json={"model_version_id": model_version_id}, headers=self.headers)
        if resp.status_code != 200:
            raise RuntimeError(f"Fairness certification failed: {resp.text}")
        return resp.json()
