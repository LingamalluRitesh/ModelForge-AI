"""
ModelForge AI Python SDK - Feature Store Module
"""

from typing import Any, Dict, List, Optional
from modelforge.exceptions import ModelForgeAPIException


class FeatureStoreModule:
    """Client interface for Feature Store catalog, online retrieval, and training set extraction."""

    def __init__(self, client):
        self.client = client

    def list_features(self, project_id: str, tag: Optional[str] = None) -> List[Dict[str, Any]]:
        params = {"project_id": project_id}
        if tag:
            params["tag"] = tag
        res = self.client._request("GET", "/feature-store/features", params=params)
        return res.get("data", [])

    def register_feature(
        self,
        project_id: str,
        name: str,
        data_type: str,
        transformation_logic: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        validation_rules: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "name": name,
            "data_type": data_type,
            "transformation_logic": transformation_logic,
            "description": description,
            "tags": tags or [],
            "validation_rules": validation_rules or {},
        }
        res = self.client._request("POST", f"/feature-store/features?project_id={project_id}", json=payload)
        return res.get("data", {})

    def fetch_online_features(
        self,
        entity_name: str,
        entity_ids: List[str],
        feature_names: List[str],
    ) -> List[Dict[str, Any]]:
        payload = {
            "entity_name": entity_name,
            "entity_ids": entity_ids,
            "feature_names": feature_names,
        }
        res = self.client._request("POST", "/feature-store/online-fetch", json=payload)
        return res.get("data", [])
