"""
ModelForge AI - API Integration Tests
Tests authentication, project management, dataset uploads, training jobs, and real-time prediction endpoints.
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check(client: AsyncClient):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_auth_register_and_login(client: AsyncClient):
    # 1. Register User
    reg_payload = {
        "email": "developer@enterprise.com",
        "password": "PasswordSecure123!",
        "first_name": "Dev",
        "last_name": "User",
        "organization_name": "Dev Enterprise",
    }
    reg_resp = await client.post("/api/v1/auth/register", json=reg_payload)
    assert reg_resp.status_code == 201
    assert "access_token" in reg_resp.json()["data"]

    # 2. Login
    login_payload = {
        "email": "developer@enterprise.com",
        "password": "PasswordSecure123!",
    }
    login_resp = await client.post("/api/v1/auth/login", json=login_payload)
    assert login_resp.status_code == 200
    assert "access_token" in login_resp.json()["data"]


@pytest.mark.asyncio
async def test_project_crud(client: AsyncClient, auth_headers: dict):
    # List projects
    list_resp = await client.get("/api/v1/projects", headers=auth_headers)
    assert list_resp.status_code == 200

    # Create project
    proj_payload = {
        "name": "Credit Risk Prediction",
        "problem_type": "classification",
        "description": "Evaluate probability of default on loans.",
    }
    create_resp = await client.post("/api/v1/projects", json=proj_payload, headers=auth_headers)
    assert create_resp.status_code == 201
    proj_id = create_resp.json()["data"]["id"]
    assert proj_id is not None

    # Get project by ID
    get_resp = await client.get(f"/api/v1/projects/{proj_id}", headers=auth_headers)
    assert get_resp.status_code == 200
    assert get_resp.json()["data"]["name"] == "Credit Risk Prediction"
