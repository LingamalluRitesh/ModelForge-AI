"""
ModelForge AI SDK - Explainability, SHAP & Integrated Gradients
"""

from typing import Any, Dict, List, Optional, Union
import httpx


class ExplainabilityClient:
    """SDK client for fetching TreeSHAP, KernelSHAP, and Integrated Gradients feature attributions."""

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.headers = {"X-API-Key": api_key} if api_key else {}

    def get_local_shap(self, model_version_id: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """Compute local SHAP waterfall feature attributions for a single inference instance."""
        url = f"{self.base_url}/api/v1/explainability/shap/local"
        payload = {
            "model_version_id": model_version_id,
            "features": features,
        }
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=payload, headers=self.headers)
            resp.raise_for_status()
            return resp.json()

    def get_global_importance(self, model_version_id: str) -> Dict[str, Any]:
        """Fetch precomputed global mean absolute SHAP feature importances."""
        url = f"{self.base_url}/api/v1/explainability/shap/global?model_version_id={model_version_id}"
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(url, headers=self.headers)
            resp.raise_for_status()
            return resp.json()
