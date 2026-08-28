"""
ModelForge AI Python SDK: Causal Inference & Uplift Client
Client interfaces for Double Machine Learning, Propensity Score Matching, and Heterogeneous Treatment Effects (CATE).
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
import requests


class CausalInferenceClient:
    """Client for executing observational causal inference and uplift modeling."""

    def __init__(self, base_url: str = "http://localhost:8000/api/v1", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}" if api_key else "",
        }

    def estimate_ate(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        confounder_cols: List[str],
        method: str = "dml",
    ) -> Dict[str, Any]:
        """Estimate Average Treatment Effect (ATE) with robust standard errors and confidence intervals."""
        payload = {
            "records": data.to_dict(orient="records"),
            "treatment_column": treatment_col,
            "outcome_column": outcome_col,
            "confounders": confounder_cols,
            "method": method,
        }

        resp = requests.post(f"{self.base_url}/causal/estimate", json=payload, headers=self.headers)
        if resp.status_code != 200:
            raise RuntimeError(f"Causal estimation failed: {resp.text}")

        return resp.json()
