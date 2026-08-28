"""
ModelForge AI - Projects & Organizations API Endpoints
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.deps import get_db, get_current_user, require_permission
from app.models.user import User
from app.models.project import Project
from app.schemas.project import ProjectCreate, ProjectUpdate, ProjectResponse
from app.schemas.organization import OrganizationCreate, OrganizationResponse
from app.schemas.common import APIResponse
from app.services.project_service import ProjectService, OrganizationService

router = APIRouter(prefix="/projects", tags=["Project Management"])
org_router = APIRouter(prefix="/organizations", tags=["Organization Management"])


# Projects
@router.get("", response_model=APIResponse[List[ProjectResponse]])
async def list_projects(
    organization_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List ML projects belonging to organization."""
    proj_svc = ProjectService(db)
    org_id = organization_id
    if not org_id:
        # Pick user's first org
        org_svc = OrganizationService(db)
        orgs = await org_svc.get_user_organizations(current_user.id)
        if orgs:
            org_id = orgs[0].id
        else:
            return APIResponse(data=[])

    projects = await proj_svc.list_projects(org_id)
    return APIResponse(data=[ProjectResponse.model_validate(p) for p in projects])


@router.post("", response_model=APIResponse[ProjectResponse], status_code=status.HTTP_201_CREATED)
async def create_project(
    payload: ProjectCreate,
    organization_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new machine learning project."""
    org_svc = OrganizationService(db)
    orgs = await org_svc.get_user_organizations(current_user.id)
    org_id = organization_id or (orgs[0].id if orgs else None)
    if not org_id:
        # Create default org if none
        new_org = await org_svc.create_organization(current_user.id, OrganizationCreate(name="Default Org"))
        org_id = new_org.id

    proj_svc = ProjectService(db)
    project = await proj_svc.create_project(org_id, current_user.id, payload)
    return APIResponse(data=ProjectResponse.model_validate(project), message="Project created successfully.")


@router.get("/{project_id}", response_model=APIResponse[ProjectResponse])
async def get_project(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get project details by ID."""
    proj_svc = ProjectService(db)
    project = await proj_svc.get_project_by_id(project_id)
    return APIResponse(data=ProjectResponse.model_validate(project))


# Organizations
@org_router.get("", response_model=APIResponse[List[OrganizationResponse]])
async def list_organizations(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List organizations the user is a member of."""
    org_svc = OrganizationService(db)
    orgs = await org_svc.get_user_organizations(current_user.id)
    return APIResponse(data=[OrganizationResponse.model_validate(o) for o in orgs])


@org_router.post("", response_model=APIResponse[OrganizationResponse], status_code=status.HTTP_201_CREATED)
async def create_organization(
    payload: OrganizationCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new enterprise organization."""
    org_svc = OrganizationService(db)
    org = await org_svc.create_organization(current_user.id, payload)
    return APIResponse(data=OrganizationResponse.model_validate(org), message="Organization created successfully.")
