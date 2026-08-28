"""
ModelForge AI - Pytest Test Configuration & Shared Fixtures
Provides in-memory async SQLite test sessions, FastAPI TestClient, and mock JWT tokens.
"""

import sys
from pathlib import Path

# Ensure backend directory is first in sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import asyncio
from typing import AsyncGenerator, Generator
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.core.database import Base, get_async_db
from app.api.deps import get_db
from app.core.security import create_access_token, get_password_hash
from app.models.user import User, Role, RoleEnum
from app.models.organization import Organization, OrganizationMember
from app.models.project import Project
from main import app

# In-memory SQLite async engine for test isolation
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(
    TEST_DATABASE_URL,
    echo=False,
    future=True,
)

TestAsyncSessionLocal = async_sessionmaker(
    bind=test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a fresh database schema for each test function."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with TestAsyncSessionLocal() as session:
        yield session

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture(scope="function")
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """FastAPI Async HTTP TestClient with overridden DB dependency."""
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_async_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest_asyncio.fixture(scope="function")
async def auth_headers(db_session: AsyncSession) -> dict:
    """Create a test Super Admin user and return authorization bearer headers."""
    super_role = Role(name=RoleEnum.SUPER_ADMIN, display_name="Super Admin", is_system_role=True)
    db_session.add(super_role)
    await db_session.flush()

    user = User(
        email="test_admin@modelforge.ai",
        hashed_password=get_password_hash("TestPassword123!"),
        first_name="Test",
        last_name="Admin",
        is_active=True,
        is_superuser=True,
        is_verified=True,
    )
    user.roles.append(super_role)
    db_session.add(user)
    await db_session.flush()

    org = Organization(
        name="Test Organization",
        slug="test-org",
        plan="enterprise",
    )
    db_session.add(org)
    await db_session.flush()

    membership = OrganizationMember(
        organization_id=org.id,
        user_id=user.id,
        role_id=super_role.id,
        is_owner=True,
    )
    db_session.add(membership)
    await db_session.commit()

    token = create_access_token(
        subject=user.id,
        claims={"email": user.email, "roles": [RoleEnum.SUPER_ADMIN], "organization_id": org.id},
    )

    return {"Authorization": f"Bearer {token}"}
