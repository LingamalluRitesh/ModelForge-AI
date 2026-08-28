"""
ModelForge AI SDK - Statistical Data & Concept Drift Client
"""

from typing import Any, Dict, List, Optional, Union
import httpx


class DriftClient:
    """SDK client for monitoring PSI, KS-test, Wasserstein, and MMD multivariate drift metrics."""

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.headers = {"X-API-Key": api_key} if api_key else {}

    def calculate_drift(
        self,
        baseline_dataset_id: str,
        target_dataset_id: str,
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Trigger offline drift computation between baseline and target datasets."""
        url = f"{self.base_url}/api/v1/monitoring/drift/compute"
        payload = {
            "baseline_dataset_id": baseline_dataset_id,
            "target_dataset_id": target_dataset_id,
            "numerical_features": numerical_features or [],
            "categorical_features": categorical_features or [],
        }
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, json=payload, headers=self.headers)
            resp.raise_for_status()
            return resp.json()
