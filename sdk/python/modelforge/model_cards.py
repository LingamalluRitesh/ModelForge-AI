"""
ModelForge AI Python SDK: Model Cards & Governance Client
"""

from typing import Any, Dict, Optional
import requests


class ModelCardClient:
    def __init__(self, base_url: str = "http://localhost:8000/api/v1", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}" if api_key else ""}

    def get_model_card(self, model_version_id: str) -> Dict[str, Any]:
        resp = requests.get(f"{self.base_url}/models/{model_version_id}/card", headers=self.headers)
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to fetch model card: {resp.text}")
        return resp.json()
