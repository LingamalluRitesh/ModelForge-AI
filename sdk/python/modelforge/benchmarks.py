"""
ModelForge AI Python SDK: Benchmarks Client
"""

from typing import Any, Dict, List, Optional
import requests


class BenchmarksClient:
    def __init__(self, base_url: str = "http://localhost:8000/api/v1", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}" if api_key else ""}

    def get_leaderboard(self, project_id: str) -> List[Dict[str, Any]]:
        resp = requests.get(f"{self.base_url}/benchmarks/projects/{project_id}/leaderboard", headers=self.headers)
        if resp.status_code != 200:
            raise RuntimeError(f"Leaderboard fetch failed: {resp.text}")
        return resp.json()
