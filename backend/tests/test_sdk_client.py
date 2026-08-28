"""
ModelForge AI - Python SDK Unit Tests
"""

import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock

# Add sdk to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "sdk" / "python"))

from modelforge.client import ModelForgeClient, ModelForgeError, AuthenticationError
from modelforge.projects import ProjectManager
from modelforge.datasets import DatasetManager


def test_sdk_client_initialization():
    client = ModelForgeClient(api_key="mf_live_testkey123", base_url="http://localhost:8000/api/v1")
    assert client.api_key == "mf_live_testkey123"
    assert client.base_url == "http://localhost:8000/api/v1"
    assert client.projects is not None
    assert client.datasets is not None
    assert client.experiments is not None
    assert client.models is not None
    assert client.deployments is not None
    assert client.predictions is not None
    assert client.monitoring is not None
    assert client.pipelines is not None


def test_sdk_client_request_mock():
    client = ModelForgeClient(api_key="mf_live_testkey123")
    client.session.request = MagicMock()
    
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "success", "data": [{"id": "p1", "name": "Fraud Detection"}]}
    client.session.request.return_value = mock_resp

    projects = client.projects.list()
    assert len(projects) == 1
    assert projects[0].id == "p1"
    assert projects[0].name == "Fraud Detection"
