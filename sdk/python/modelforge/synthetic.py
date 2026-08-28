"""
ModelForge AI Python SDK: Synthetic Data & Privacy Client
Client interfaces for training Conditional GANs, Copulas, and evaluating Privacy Bounds.
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
import requests


class SyntheticDataClient:
    """Client for generating high-fidelity synthetic tabular datasets with Differential Privacy guarantees."""

    def __init__(self, base_url: str = "http://localhost:8000/api/v1", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}" if api_key else "",
        }

    def generate(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]]],
        algorithm: str = "ctgan",
        num_samples: int = 1000,
        continuous_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Generate privacy-safe synthetic DataFrame."""
        if isinstance(data, pd.DataFrame):
            records = data.to_dict(orient="records")
        else:
            records = data

        payload = {
            "source_records": records,
            "algorithm": algorithm,
            "num_samples": num_samples,
            "continuous_columns": continuous_columns,
        }

        resp = requests.post(f"{self.base_url}/synthetic/generate", json=payload, headers=self.headers)
        if resp.status_code != 200:
            raise RuntimeError(f"Synthetic generation failed: {resp.text}")

        res_json = resp.json()
        return pd.DataFrame(res_json["records"])
