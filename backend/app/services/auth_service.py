"""
ModelForge AI - Auth & Security Domain Service
Handles User Registration, Login, Token Issuance/Refresh, API Keys, MFA, and Security Audits.
"""

from datetime import datetime, timezone, timedelta
import secrets
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.config import settings
from app.core.security import (
    verify_password, get_password_hash, create_access_token, create_refresh_token,
    decode_token, generate_api_key, verify_api_key
)
from app.core.exceptions import (
    AuthenticationException, EntityAlreadyExistsException, EntityNotFoundException, ValidationException
)
from app.models.user import User, Role, RoleEnum, APIKey, LoginHistory
from app.models.organization import Organization, OrganizationMember
from app.schemas.auth import (
    RegisterRequest, LoginRequest, TokenResponse, APIKeyCreate, APIKeyResponse
)


class AuthService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def register_user(self, payload: RegisterRequest) -> Tuple[User, TokenResponse]:
        """Register a new user, create default org if specified, and issue access tokens."""
        # Check existing user
        query = select(User).where(User.email == payload.email)
        res = await self.session.execute(query)
        if res.scalar_one_or_none():
            raise EntityAlreadyExistsException("User", payload.email)

        # Fetch default role (ML Engineer)
        query = select(Role).where(Role.name == RoleEnum.ML_ENGINEER)
        res = await self.session.execute(query)
        ml_role = res.scalar_one_or_none()

        user = User(
            email=payload.email,
            hashed_password=get_password_hash(payload.password),
            first_name=payload.first_name,
            last_name=payload.last_name,
            is_active=True,
            is_verified=True,
        )
        if ml_role:
            user.roles.append(ml_role)
        self.session.add(user)
        await self.session.flush()

        # Handle Organization
        org_name = payload.organization_name or f"{payload.first_name}'s Workspace"
        org_slug = f"{payload.first_name.lower()}-{secrets.token_hex(4)}"

        # Fetch Org Admin role
        query = select(Role).where(Role.name == RoleEnum.ORG_ADMIN)
        res = await self.session.execute(query)
        org_admin_role = res.scalar_one_or_none() or ml_role

        org = Organization(
            name=org_name,
            slug=org_slug,
            plan="developer",
            max_projects=20,
            max_storage_gb=100,
        )
        self.session.add(org)
        await self.session.flush()

        membership = OrganizationMember(
            organization_id=org.id,
            user_id=user.id,
            role_id=org_admin_role.id,
            is_owner=True,
        )
        self.session.add(membership)
        await self.session.commit()

        # Generate tokens
        token_resp = self._generate_token_response(user, org.id)
        return user, token_resp

    async def authenticate_user(self, payload: LoginRequest, ip_address: str = "127.0.0.1", user_agent: str = "") -> TokenResponse:
        """Authenticate user credentials, check lockout, and issue tokens."""
        query = select(User).where(User.email == payload.email)
        res = await self.session.execute(query)
        user = res.scalar_one_or_none()

        if not user or not verify_password(payload.password, user.hashed_password):
            if user:
                user.failed_login_attempts += 1
                if user.failed_login_attempts >= 5:
                    user.locked_until = datetime.now(timezone.utc) + timedelta(minutes=15)
                # Log failed attempt
                log_entry = LoginHistory(
                    user_id=user.id,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    success=False,
                    failure_reason="Invalid credentials",
                )
                self.session.add(log_entry)
                await self.session.commit()
            raise AuthenticationException("Invalid email or password.")

        if not user.is_active:
            raise AuthenticationException("User account is deactivated. Contact administrator.")

        if user.locked_until and user.locked_until > datetime.now(timezone.utc):
            raise AuthenticationException("Account is temporarily locked due to failed attempts. Please retry later.")

        # Reset failed attempts and update last login
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login_at = datetime.now(timezone.utc)

        # Fetch user's primary organization ID
        query = select(OrganizationMember).where(OrganizationMember.user_id == user.id)
        res = await self.session.execute(query)
        membership = res.scalars().first()
        org_id = membership.organization_id if membership else None

        # Log successful login
        log_entry = LoginHistory(
            user_id=user.id,
            ip_address=ip_address,
            user_agent=user_agent,
            success=True,
        )
        self.session.add(log_entry)
        await self.session.commit()

        return self._generate_token_response(user, org_id)

    async def refresh_access_token(self, refresh_token: str) -> TokenResponse:
        """Issue new access token from valid refresh token."""
        try:
            payload = decode_token(refresh_token)
            if payload.get("type") != "refresh":
                raise AuthenticationException("Invalid token type for refresh.")
            user_id = payload.get("sub")
        except Exception:
            raise AuthenticationException("Invalid or expired refresh token.")

        query = select(User).where(User.id == user_id)
        res = await self.session.execute(query)
        user = res.scalar_one_or_none()

        if not user or not user.is_active:
            raise AuthenticationException("User not found or inactive.")

        query = select(OrganizationMember).where(OrganizationMember.user_id == user.id)
        res = await self.session.execute(query)
        membership = res.scalars().first()
        org_id = membership.organization_id if membership else None

        return self._generate_token_response(user, org_id)

    async def create_api_key(self, user_id: str, payload: APIKeyCreate) -> Tuple[APIKey, str]:
        """Generate a new secure hashed API key for programmatic access."""
        raw_key, prefix, key_hash = generate_api_key(prefix="mf_live_")
        expires_at = None
        if payload.expires_in_days:
            expires_at = datetime.now(timezone.utc) + timedelta(days=payload.expires_in_days)

        api_key = APIKey(
            user_id=user_id,
            name=payload.name,
            key_prefix=prefix,
            key_hash=key_hash,
            scopes=payload.scopes or [],
            expires_at=expires_at,
        )
        self.session.add(api_key)
        await self.session.commit()
        return api_key, raw_key

    async def verify_incoming_api_key(self, raw_key: str) -> Optional[User]:
        """Authenticate user by incoming Bearer API Key."""
        import hashlib
        key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        query = select(APIKey).where(APIKey.key_hash == key_hash, APIKey.is_active == True)
        res = await self.session.execute(query)
        api_key = res.scalar_one_or_none()

        if not api_key:
            return None

        if api_key.expires_at and api_key.expires_at < datetime.now(timezone.utc):
            return None

        api_key.last_used_at = datetime.now(timezone.utc)
        await self.session.commit()

        query = select(User).where(User.id == api_key.user_id)
        res = await self.session.execute(query)
        return res.scalar_one_or_none()

    def _generate_token_response(self, user: User, organization_id: Optional[str]) -> TokenResponse:
        roles_list = [r.name for r in user.roles]
        claims = {
            "email": user.email,
            "roles": roles_list,
            "is_superuser": user.is_superuser,
            "organization_id": organization_id,
        }
        access_token = create_access_token(subject=user.id, claims=claims)
        refresh_token = create_refresh_token(subject=user.id)

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in_seconds=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user_id=user.id,
            email=user.email,
            roles=roles_list,
            organization_id=organization_id,
        )
