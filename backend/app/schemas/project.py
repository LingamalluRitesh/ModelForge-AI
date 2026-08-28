"""
ModelForge AI - Project Schemas
"""

from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
from app.schemas.user import UserResponse, RoleResponse


class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None
    problem_type: str = "classification"  # classification, regression, clustering, deep_learning
    tags: Optional[List[str]] = None
    settings: Optional[dict] = None


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    problem_type: Optional[str] = None
    tags: Optional[List[str]] = None
    is_archived: Optional[bool] = None
    settings: Optional[dict] = None


class ProjectMemberCreate(BaseModel):
    user_id: str
    role_id: str


class ProjectMemberResponse(BaseModel):
    id: str
    user: UserResponse
    role: RoleResponse
    created_at: datetime

    model_config = {"from_attributes": True}


class ProjectResponse(BaseModel):
    id: str
    organization_id: str
    name: str
    slug: str
    description: Optional[str] = None
    problem_type: str
    tags: Optional[List[str]] = None
    is_archived: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
