"""
ModelForge AI Python SDK: Real-Time Streaming Ingestion Client
"""

from typing import Any, Dict, List, Optional
import requests


class StreamingTelemetryClient:
    def __init__(self, base_url: str = "http://localhost:8000/api/v1", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}" if api_key else ""}

    def ingest_streaming_records(self, deployment_id: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        resp = requests.post(
            f"{self.base_url}/monitoring/telemetry",
            json={"deployment_id": deployment_id, "records": records},
            headers=self.headers,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Streaming ingestion failed: {resp.text}")
        return resp.json()
