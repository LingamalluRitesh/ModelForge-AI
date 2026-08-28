"""
ModelForge AI - Official Python Client SDK
High-performance synchronous and asynchronous Python client for enterprise ML operations.
"""

import os
import time
import json
import logging
from typing import Any, Dict, List, Optional, Union
import requests

logger = logging.getLogger("modelforge-sdk")


class ModelForgeError(Exception):
    """Base exception for ModelForge SDK errors."""
    pass


class AuthenticationError(ModelForgeError):
    """Raised when API credentials or JWT tokens are invalid."""
    pass


class ModelForgeClient:
    """
    Primary client interface for interacting with the ModelForge AI platform.
    
    Example:
    ```python
    from modelforge import ModelForgeClient

    client = ModelForgeClient(api_key="mf_live_...")
    project = client.projects.get("fraud-detection")
    dataset = client.datasets.upload(project_id=project.id, file_path="data.csv")
    run = client.experiments.train(
        project_id=project.id,
        algorithm="xgboost",
        dataset_id=dataset.id,
        target_column="is_fraud"
    )
    print(f"Model trained! F1 Score: {run.metrics['f1']}")
    ```
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
    ):
        self.api_key = api_key or os.environ.get("MODELFORGE_API_KEY")
        self.base_url = (base_url or os.environ.get("MODELFORGE_BASE_URL") or "http://localhost:8000/api/v1").rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

        if self.api_key:
            self.session.headers.update({"X-API-Key": self.api_key})
        
        self.session.headers.update({
            "User-Agent": "ModelForge-Python-SDK/1.0.0",
            "Accept": "application/json",
        })

        # Lazy-initialized resource managers
        self._projects = None
        self._datasets = None
        self._features = None
        self._experiments = None
        self._models = None
        self._deployments = None
        self._predictions = None
        self._monitoring = None
        self._pipelines = None
        self._causal = None
        self._federated = None
        self._governance = None
        self._security = None

    def request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Execute HTTP request with automatic error mapping and retries."""
        url = f"{self.base_url}/{path.lstrip('/')}"
        req_headers = {}
        if headers:
            req_headers.update(headers)

        for attempt in range(3):
            try:
                resp = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_data,
                    files=files,
                    headers=req_headers,
                    timeout=self.timeout,
                )
                if resp.status_code == 401:
                    raise AuthenticationError("Invalid or expired API Key / Token.")
                if resp.status_code >= 400:
                    try:
                        err_json = resp.json()
                        msg = err_json.get("error", {}).get("message") or resp.text
                    except Exception:
                        msg = resp.text
                    raise ModelForgeError(f"API Error ({resp.status_code}): {msg}")

                return resp.json()
            except (requests.ConnectionError, requests.Timeout) as conn_err:
                if attempt == 2:
                    raise ModelForgeError(f"Network communication failed after 3 attempts: {conn_err}")
                time.sleep(1.0 * (2 ** attempt))

    def _request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        """Internal helper forwarding to request()."""
        json_data = kwargs.get("json") or kwargs.get("json_data")
        params = kwargs.get("params")
        files = kwargs.get("files")
        headers = kwargs.get("headers")
        return self.request(method=method, path=path, params=params, json_data=json_data, files=files, headers=headers)

    @property
    def projects(self):
        if self._projects is None:
            from modelforge.projects import ProjectManager
            self._projects = ProjectManager(self)
        return self._projects

    @property
    def datasets(self):
        if self._datasets is None:
            from modelforge.datasets import DatasetManager
            self._datasets = DatasetManager(self)
        return self._datasets

    @property
    def features(self):
        if self._features is None:
            from modelforge.feature_store import FeatureStoreModule
            self._features = FeatureStoreModule(self)
        return self._features

    @property
    def experiments(self):
        if self._experiments is None:
            from modelforge.experiments import ExperimentManager
            self._experiments = ExperimentManager(self)
        return self._experiments

    @property
    def models(self):
        if self._models is None:
            from modelforge.models import ModelRegistryManager
            self._models = ModelRegistryManager(self)
        return self._models

    @property
    def deployments(self):
        if self._deployments is None:
            from modelforge.deployments import DeploymentManager
            self._deployments = DeploymentManager(self)
        return self._deployments

    @property
    def predictions(self):
        if self._predictions is None:
            from modelforge.deployments import PredictionManager
            self._predictions = PredictionManager(self)
        return self._predictions

    @property
    def monitoring(self):
        if self._monitoring is None:
            from modelforge.monitoring import MonitoringManager
            self._monitoring = MonitoringManager(self)
        return self._monitoring

    @property
    def pipelines(self):
        if self._pipelines is None:
            from modelforge.pipelines import PipelineManager
            self._pipelines = PipelineManager(self)
        return self._pipelines

    @property
    def causal(self):
        if self._causal is None:
            from modelforge.causal import CausalModule
            self._causal = CausalModule(self)
        return self._causal

    @property
    def federated(self):
        if self._federated is None:
            from modelforge.federated import FederatedModule
            self._federated = FederatedModule(self)
        return self._federated

    @property
    def governance(self):
        if self._governance is None:
            from modelforge.governance import GovernanceModule
            self._governance = GovernanceModule(self)
        return self._governance

    @property
    def security(self):
        if self._security is None:
            from modelforge.security import SecurityModule
            self._security = SecurityModule(self)
        return self._security
