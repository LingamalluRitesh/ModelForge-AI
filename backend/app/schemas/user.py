"""
ModelForge AI - User & Role Management Schemas
"""

from typing import List, Optional
from pydantic import BaseModel, EmailStr
from datetime import datetime


class PermissionResponse(BaseModel):
    id: str
    resource: str
    action: str
    description: Optional[str] = None

    model_config = {"from_attributes": True}


class RoleResponse(BaseModel):
    id: str
    name: str
    display_name: str
    description: Optional[str] = None
    is_system_role: bool
    permissions: List[PermissionResponse] = []

    model_config = {"from_attributes": True}


class UserResponse(BaseModel):
    id: str
    email: EmailStr
    first_name: str
    last_name: str
    full_name: str
    is_active: bool
    is_superuser: bool
    is_verified: bool
    mfa_enabled: bool
    roles: List[RoleResponse] = []
    created_at: datetime
    last_login_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    first_name: str
    last_name: str
    role_ids: List[str] = []
    is_active: bool = True


class UserUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    password: Optional[str] = None
    is_active: Optional[bool] = None
    role_ids: Optional[List[str]] = None
