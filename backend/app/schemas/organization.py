"""
ModelForge AI - Organization & Team Schemas
"""

from typing import List, Optional
from pydantic import BaseModel, EmailStr
from datetime import datetime
from app.schemas.user import UserResponse, RoleResponse


class OrganizationCreate(BaseModel):
    name: str
    description: Optional[str] = None


class OrganizationUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    max_projects: Optional[int] = None
    max_storage_gb: Optional[int] = None
    settings: Optional[dict] = None


class OrganizationMemberResponse(BaseModel):
    id: str
    user: UserResponse
    role: RoleResponse
    is_owner: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class OrganizationResponse(BaseModel):
    id: str
    name: str
    slug: str
    description: Optional[str] = None
    plan: str
    max_projects: int
    max_storage_gb: int
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class TeamCreate(BaseModel):
    name: str
    description: Optional[str] = None


class TeamResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    organization_id: str
    created_at: datetime

    model_config = {"from_attributes": True}


class InvitationCreate(BaseModel):
    email: EmailStr
    role_id: str


class InvitationResponse(BaseModel):
    id: str
    email: EmailStr
    role_id: str
    is_accepted: bool
    expires_at: datetime
    created_at: datetime

    model_config = {"from_attributes": True}
