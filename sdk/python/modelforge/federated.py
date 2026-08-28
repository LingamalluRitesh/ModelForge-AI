"""
ModelForge AI Python SDK: Federated Learning Client
Client interfaces for Federated Averaging (FedAvg), Differential Privacy noise addition, and decentralized model updates.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import requests


class FederatedClient:
    """Client for local edge node participation in Federated Learning rounds."""

    def __init__(self, base_url: str = "http://localhost:8000/api/v1", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}" if api_key else "",
        }

    def submit_round_update(
        self,
        session_id: str,
        client_id: str,
        round_number: int,
        weights: List[np.ndarray],
        num_samples: int,
        loss: float,
    ) -> Dict[str, Any]:
        """Submit gradient/weight updates to central parameter server."""
        serialized_weights = [w.tolist() for w in weights]
        payload = {
            "session_id": session_id,
            "client_id": client_id,
            "round_number": round_number,
            "weights": serialized_weights,
            "num_samples": num_samples,
            "loss": loss,
        }

        resp = requests.post(f"{self.base_url}/federated/submit", json=payload, headers=self.headers)
        if resp.status_code != 200:
            raise RuntimeError(f"Federated update submission failed: {resp.text}")

        return resp.json()
