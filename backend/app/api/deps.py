"""
ModelForge AI - FastAPI Dependency Injection Suite
Provides authenticated user extraction, API key verification, granular RBAC permission enforcement,
database session injection, and rate limiting.
"""

from typing import AsyncGenerator, Callable, List, Optional
from fastapi import Depends, HTTPException, Header, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_async_db
from app.core.security import decode_token, verify_api_key
from app.core.exceptions import AuthenticationException, AuthorizationException
from app.core.rate_limiter import RateLimiter
from app.models.user import User, RoleEnum
from app.services.auth_service import AuthService

security_bearer = HTTPBearer(auto_error=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Provide async database session."""
    async for session in get_async_db():
        yield session


async def get_current_user(
    request: Request,
    auth_header: Optional[HTTPAuthorizationCredentials] = Depends(security_bearer),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Authenticate incoming request using either JWT Bearer token or X-API-Key."""
    # 1. Check API Key authentication
    if x_api_key:
        auth_svc = AuthService(db)
        user = await auth_svc.verify_incoming_api_key(x_api_key)
        if not user:
            raise AuthenticationException("Invalid or expired API Key.")
        # Apply rate limiting per API Key
        await RateLimiter.check_rate_limit(f"apikey_{user.id}")
        return user

    # 2. Check JWT Bearer token
    if not auth_header or not auth_header.credentials:
        raise AuthenticationException("Missing authentication credentials (JWT Bearer or X-API-Key required).")

    token = auth_header.credentials
    try:
        payload = decode_token(token)
        user_id = payload.get("sub")
        if not user_id:
            raise AuthenticationException("Token payload missing subject.")
    except Exception as e:
        raise AuthenticationException(f"Invalid authentication token: {str(e)}")

    query = select(User).where(User.id == user_id)
    res = await db.execute(query)
    user = res.scalar_one_or_none()

    if not user:
        raise AuthenticationException("User account not found.")
    if not user.is_active:
        raise AuthenticationException("User account is inactive.")

    # Rate limiting per user
    await RateLimiter.check_rate_limit(f"user_{user.id}")
    return user


async def get_current_active_superuser(
    current_user: User = Depends(get_current_user),
) -> User:
    """Ensure authenticated user has SUPER_ADMIN or superuser status."""
    is_super = current_user.is_superuser or any(r.name == RoleEnum.SUPER_ADMIN for r in current_user.roles)
    if not is_super:
        raise AuthorizationException("This action requires Super Admin privileges.")
    return current_user


def require_permission(resource: str, action: str) -> Callable:
    """Decorator / dependency factory for granular RBAC permission enforcement."""
    async def permission_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.is_superuser:
            return current_user

        user_role_names = [r.name for r in current_user.roles]
        if RoleEnum.SUPER_ADMIN in user_role_names or RoleEnum.ORG_ADMIN in user_role_names:
            return current_user

        # Check permissions attached to user roles
        for role in current_user.roles:
            for perm in role.permissions:
                if (perm.resource == resource or perm.resource == "*") and (perm.action == action or perm.action == "*"):
                    return current_user

        raise AuthorizationException(f"Permission denied: Requires permission '{resource}:{action}'.")

    return permission_checker
