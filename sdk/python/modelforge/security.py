"""
ModelForge AI Python SDK - Model Security & Penetration Testing Module
"""

from typing import Any, Dict, List, Optional
from modelforge.exceptions import ModelForgeAPIException


class SecurityModule:
    """Client interface for adversarial robustness tests and certified security bounds."""

    def __init__(self, client):
        self.client = client

    def audit_adversarial_robustness(
        self,
        model_version_id: str,
        test_dataset_version_id: str,
        target_column: str,
        epsilon: float = 0.05,
    ) -> Dict[str, Any]:
        """Execute automated FGSM adversarial perturbation attack to evaluate model vulnerability."""
        payload = {
            "model_version_id": model_version_id,
            "test_dataset_version_id": test_dataset_version_id,
            "target_column": target_column,
            "epsilon": epsilon,
        }
        res = self.client._request("POST", "/security-audit/adversarial-test", json=payload)
        return res.get("data", {})
