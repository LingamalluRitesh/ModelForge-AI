"""
ModelForge AI - Model Registry & Deployments Integration Tests
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_model_registry_flow(client: AsyncClient, auth_headers: dict):
    # 1. Create Project
    proj_resp = await client.post(
        "/api/v1/projects",
        json={"name": "Registry Test Project", "problem_type": "classification"},
        headers=auth_headers,
    )
    proj_id = proj_resp.json()["data"]["id"]

    # 2. Register Model
    model_resp = await client.post(
        f"/api/v1/registry/models?project_id={proj_id}",
        json={"name": "Churn XGBoost Model", "problem_type": "classification"},
        headers=auth_headers,
    )
    assert model_resp.status_code == 201
    model_id = model_resp.json()["data"]["id"]

    # 3. Create Version
    ver_resp = await client.post(
        f"/api/v1/registry/models/{model_id}/versions",
        json={"version_tag": "v1.0.0", "description": "Initial baseline model"},
        headers=auth_headers,
    )
    assert ver_resp.status_code == 201
    ver_id = ver_resp.json()["data"]["id"]

    # 4. Request Approval to Staging
    appr_resp = await client.post(
        f"/api/v1/registry/versions/{ver_id}/request-approval",
        json={"target_stage": "staging", "notes": "Passed all offline validation tests"},
        headers=auth_headers,
    )
    assert appr_resp.status_code == 201


@pytest.mark.asyncio
async def test_canary_deployment_routing(client: AsyncClient, auth_headers: dict):
    # 1. Create Project & Model
    proj_resp = await client.post(
        "/api/v1/projects",
        json={"name": "Deployment Test Project", "problem_type": "classification"},
        headers=auth_headers,
    )
    proj_id = proj_resp.json()["data"]["id"]

    model_resp = await client.post(
        f"/api/v1/registry/models?project_id={proj_id}",
        json={"name": "Fraud Serving Model", "problem_type": "classification"},
        headers=auth_headers,
    )
    model_id = model_resp.json()["data"]["id"]

    ver_resp = await client.post(
        f"/api/v1/registry/models/{model_id}/versions",
        json={"version_tag": "v1.0.0"},
        headers=auth_headers,
    )
    ver_id = ver_resp.json()["data"]["id"]

    # 2. Deploy Model
    deploy_resp = await client.post(
        f"/api/v1/deployments?project_id={proj_id}",
        json={
            "name": "Fraud Canary Endpoint",
            "endpoint_path": "fraud-canary-test",
            "model_version_id": ver_id,
            "strategy": "canary",
            "primary_traffic_percentage": 90.0,
            "canary_stage_percentage": 10.0,
        },
        headers=auth_headers,
    )
    assert deploy_resp.status_code == 201
    dep_id = deploy_resp.json()["data"]["id"]

    # 3. Update Canary Traffic to 50%
    canary_resp = await client.post(
        f"/api/v1/deployments/{dep_id}/canary",
        json={"canary_stage_percentage": 50.0},
        headers=auth_headers,
    )
    assert canary_resp.status_code == 200
    assert canary_resp.json()["data"]["canary_stage_percentage"] == 50.0

    # 4. Trigger Rollback
    rollback_resp = await client.post(
        f"/api/v1/deployments/{dep_id}/rollback",
        json={"reason": "Testing rollback handler"},
        headers=auth_headers,
    )
    assert rollback_resp.status_code == 200
