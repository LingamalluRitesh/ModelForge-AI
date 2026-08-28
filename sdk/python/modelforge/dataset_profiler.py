"""
ModelForge AI SDK - Dataset Profiling & Data Quality Client
"""

from typing import Any, Dict, List, Optional, Union
import httpx


class DatasetProfilerClient:
    """SDK client for automated data profiling, missingness detection, and correlation analysis."""

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.headers = {"X-API-Key": api_key} if api_key else {}

    def get_profile(self, dataset_id: str) -> Dict[str, Any]:
        """Fetch full statistical summary profile for a registered dataset."""
        url = f"{self.base_url}/api/v1/datasets/{dataset_id}/profile"
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(url, headers=self.headers)
            resp.raise_for_status()
            return resp.json()

    def run_quality_check(self, dataset_id: str, rules: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Execute enterprise data quality checks (Great Expectations compliant)."""
        url = f"{self.base_url}/api/v1/datasets/{dataset_id}/quality-check"
        payload = {"rules": rules or []}
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=payload, headers=self.headers)
            resp.raise_for_status()
            return resp.json()
