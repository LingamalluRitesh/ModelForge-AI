"""
ModelForge AI - Authentication & User Account Endpoints
"""

from typing import Any, List
from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.deps import get_db, get_current_user
from app.models.user import User, APIKey
from app.schemas.auth import (
    RegisterRequest, LoginRequest, TokenResponse, RefreshTokenRequest,
    APIKeyCreate, APIKeyResponse
)
from app.schemas.user import UserResponse
from app.schemas.common import APIResponse
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["Authentication & Identity"])


@router.post("/register", response_model=APIResponse[TokenResponse], status_code=status.HTTP_201_CREATED)
async def register(
    payload: RegisterRequest,
    db: AsyncSession = Depends(get_db),
):
    """Register a new user account and issue access tokens."""
    auth_svc = AuthService(db)
    user, tokens = await auth_svc.register_user(payload)
    return APIResponse(data=tokens, message="User registered successfully.")


@router.post("/login", response_model=APIResponse[TokenResponse])
async def login(
    payload: LoginRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Authenticate user credentials and receive JWT access/refresh tokens."""
    client_ip = request.client.host if request.client else "127.0.0.1"
    user_agent = request.headers.get("User-Agent", "")
    auth_svc = AuthService(db)
    tokens = await auth_svc.authenticate_user(payload, ip_address=client_ip, user_agent=user_agent)
    return APIResponse(data=tokens, message="Authentication successful.")


@router.post("/refresh", response_model=APIResponse[TokenResponse])
async def refresh_token(
    payload: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db),
):
    """Obtain a fresh access token using a valid refresh token."""
    auth_svc = AuthService(db)
    tokens = await auth_svc.refresh_access_token(payload.refresh_token)
    return APIResponse(data=tokens, message="Token refreshed successfully.")


@router.get("/me", response_model=APIResponse[UserResponse])
async def get_current_user_profile(
    current_user: User = Depends(get_current_user),
):
    """Return the profile and permissions of the currently authenticated user."""
    return APIResponse(data=UserResponse.model_validate(current_user))


@router.post("/api-keys", response_model=APIResponse[APIKeyResponse], status_code=status.HTTP_201_CREATED)
async def create_api_key(
    payload: APIKeyCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Generate a new API key for programmatic SDK/CLI access."""
    auth_svc = AuthService(db)
    api_key, raw_key = await auth_svc.create_api_key(current_user.id, payload)
    resp = APIKeyResponse.model_validate(api_key)
    resp.raw_key = raw_key
    return APIResponse(data=resp, message="API Key generated. Save it now; it will not be shown again.")


@router.get("/api-keys", response_model=APIResponse[List[APIKeyResponse]])
async def list_api_keys(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List active API keys for the current user."""
    query = select(APIKey).where(APIKey.user_id == current_user.id, APIKey.is_active == True)
    res = await db.execute(query)
    keys = list(res.scalars().all())
    return APIResponse(data=[APIKeyResponse.model_validate(k) for k in keys])
