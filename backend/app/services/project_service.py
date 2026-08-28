"""
ModelForge AI - Project & Organization Domain Services
"""

from typing import Any, Dict, List, Optional
import secrets
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.core.exceptions import EntityNotFoundException, EntityAlreadyExistsException, AuthorizationException
from app.models.organization import Organization, OrganizationMember, Team, TeamMember, Invitation
from app.models.project import Project, ProjectMember
from app.models.user import User, Role, RoleEnum
from app.schemas.organization import OrganizationCreate, OrganizationUpdate, TeamCreate, InvitationCreate
from app.schemas.project import ProjectCreate, ProjectUpdate, ProjectMemberCreate


class OrganizationService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_user_organizations(self, user_id: str) -> List[Organization]:
        query = select(Organization).join(OrganizationMember).where(OrganizationMember.user_id == user_id)
        res = await self.session.execute(query)
        return list(res.scalars().all())

    async def get_organization_by_id(self, org_id: str) -> Organization:
        query = select(Organization).where(Organization.id == org_id)
        res = await self.session.execute(query)
        org = res.scalar_one_or_none()
        if not org:
            raise EntityNotFoundException("Organization", org_id)
        return org

    async def create_organization(self, user_id: str, payload: OrganizationCreate) -> Organization:
        slug = f"{payload.name.lower().replace(' ', '-')}-{secrets.token_hex(3)}"
        org = Organization(
            name=payload.name,
            slug=slug,
            description=payload.description,
            plan="enterprise",
        )
        self.session.add(org)
        await self.session.flush()

        # Get Org Admin role
        query = select(Role).where(Role.name == RoleEnum.ORG_ADMIN)
        res = await self.session.execute(query)
        role = res.scalar_one_or_none()

        membership = OrganizationMember(
            organization_id=org.id,
            user_id=user_id,
            role_id=role.id if role else "",
            is_owner=True,
        )
        self.session.add(membership)
        await self.session.commit()
        return org


class ProjectService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def list_projects(self, organization_id: str) -> List[Project]:
        query = select(Project).where(Project.organization_id == organization_id, Project.is_archived == False)
        res = await self.session.execute(query)
        return list(res.scalars().all())

    async def get_project_by_id(self, project_id: str) -> Project:
        query = select(Project).where(Project.id == project_id)
        res = await self.session.execute(query)
        proj = res.scalar_one_or_none()
        if not proj:
            raise EntityNotFoundException("Project", project_id)
        return proj

    async def create_project(self, organization_id: str, user_id: str, payload: ProjectCreate) -> Project:
        slug = f"{payload.name.lower().replace(' ', '-')}-{secrets.token_hex(3)}"
        proj = Project(
            organization_id=organization_id,
            name=payload.name,
            slug=slug,
            description=payload.description,
            problem_type=payload.problem_type,
            tags=payload.tags or [],
            settings=payload.settings or {},
        )
        self.session.add(proj)
        await self.session.flush()

        # Add creator as project member with ML_ENGINEER role
        query = select(Role).where(Role.name == RoleEnum.ML_ENGINEER)
        res = await self.session.execute(query)
        role = res.scalar_one_or_none()

        if role:
            member = ProjectMember(
                project_id=proj.id,
                user_id=user_id,
                role_id=role.id,
            )
            self.session.add(member)

        await self.session.commit()
        return proj
