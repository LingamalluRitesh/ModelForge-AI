"""
ModelForge AI - Database Seeder & Initialization
Bootstraps initial roles, granular RBAC permissions, superuser account, and enterprise organization.
"""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.config import settings
from app.core.security import get_password_hash
from app.core.logging import logger
from app.models.user import User, Role, Permission, RoleEnum
from app.models.organization import Organization, OrganizationMember


# Granular system permissions across all platform resources
DEFAULT_PERMISSIONS = [
    # Organizations & Projects
    ("organization", "create", "Create new organizations"),
    ("organization", "read", "View organization settings and members"),
    ("organization", "update", "Update organization settings"),
    ("organization", "delete", "Delete organization"),
    ("project", "create", "Create new ML projects"),
    ("project", "read", "View ML projects"),
    ("project", "update", "Update ML project configurations"),
    ("project", "delete", "Archive or delete ML projects"),

    # Datasets & Features
    ("dataset", "create", "Upload or ingest new datasets"),
    ("dataset", "read", "Preview and download datasets"),
    ("dataset", "validate", "Run data quality analysis"),
    ("dataset", "delete", "Delete dataset versions"),
    ("feature", "create", "Register and transform features in Feature Store"),
    ("feature", "read", "Retrieve features for training and serving"),
    ("feature", "delete", "Deprecate or delete features"),

    # Experiments & Training
    ("experiment", "create", "Create experiments and log runs"),
    ("experiment", "read", "View runs, metrics, and hyperparameter plots"),
    ("training", "execute", "Launch model training and AutoML jobs"),
    ("training", "cancel", "Stop active training runs"),

    # Registry & Approvals
    ("model_registry", "register", "Register trained models to registry"),
    ("model_registry", "read", "View model registry versions and artifacts"),
    ("model_registry", "approve", "Approve or reject models for staging/production"),
    ("model_registry", "archive", "Archive model versions"),

    # Deployments & Inference
    ("deployment", "create", "Deploy models to staging or production"),
    ("deployment", "read", "View deployment status and endpoints"),
    ("deployment", "rollback", "Trigger deployment rollback"),
    ("deployment", "scale", "Scale inference replicas"),
    ("inference", "predict", "Execute real-time or batch predictions"),

    # Monitoring, Drift & Retraining
    ("monitoring", "read", "View live latency, error rates, and ML metrics"),
    ("drift", "detect", "Run data and concept drift analysis"),
    ("retraining", "trigger", "Trigger automated retraining pipelines"),

    # Pipelines & Governance
    ("pipeline", "create", "Build and schedule visual DAG pipelines"),
    ("pipeline", "execute", "Trigger manual pipeline execution"),
    ("alert", "manage", "Configure alert rules and acknowledge notifications"),
    ("audit", "read", "View immutable audit logs"),
]

# Role to permission mapping
ROLE_PERMISSION_MAP = {
    RoleEnum.SUPER_ADMIN: ["*:*"], # All permissions
    RoleEnum.ORG_ADMIN: [
        "organization:*", "project:*", "dataset:*", "feature:*", "experiment:*",
        "training:*", "model_registry:*", "deployment:*", "inference:*",
        "monitoring:*", "drift:*", "retraining:*", "pipeline:*", "alert:*", "audit:*"
    ],
    RoleEnum.ML_ENGINEER: [
        "project:read", "dataset:*", "feature:*", "experiment:*", "training:*",
        "model_registry:register", "model_registry:read", "deployment:*",
        "inference:*", "monitoring:*", "drift:*", "retraining:*", "pipeline:*", "alert:manage"
    ],
    RoleEnum.DATA_SCIENTIST: [
        "project:read", "dataset:*", "feature:*", "experiment:*", "training:*",
        "model_registry:register", "model_registry:read", "inference:*",
        "monitoring:read", "drift:*", "retraining:trigger", "pipeline:*"
    ],
    RoleEnum.DATA_ENGINEER: [
        "project:read", "dataset:*", "feature:*", "pipeline:*", "inference:predict", "monitoring:read"
    ],
    RoleEnum.MODEL_REVIEWER: [
        "project:read", "dataset:read", "experiment:read", "model_registry:read",
        "model_registry:approve", "monitoring:read", "drift:detect", "audit:read"
    ],
    RoleEnum.BUSINESS_ANALYST: [
        "project:read", "dataset:read", "experiment:read", "model_registry:read",
        "deployment:read", "monitoring:read", "inference:predict"
    ],
    RoleEnum.VIEWER: [
        "project:read", "dataset:read", "experiment:read", "model_registry:read",
        "deployment:read", "monitoring:read"
    ],
}


async def init_db_data(session: AsyncSession) -> None:
    """Seed base permissions, default roles, super admin, and initial enterprise org."""
    logger.info("Initializing database base data...")

    # 1. Create permissions
    permission_objects = {}
    for resource, action, desc in DEFAULT_PERMISSIONS:
        query = select(Permission).where(Permission.resource == resource, Permission.action == action)
        res = await session.execute(query)
        perm = res.scalar_one_or_none()
        if not perm:
            perm = Permission(resource=resource, action=action, description=desc)
            session.add(perm)
            await session.flush()
        permission_objects[f"{resource}:{action}"] = perm

    # 2. Create standard roles
    role_objects = {}
    for role_name in [
        RoleEnum.SUPER_ADMIN, RoleEnum.ORG_ADMIN, RoleEnum.ML_ENGINEER,
        RoleEnum.DATA_SCIENTIST, RoleEnum.DATA_ENGINEER, RoleEnum.MODEL_REVIEWER,
        RoleEnum.BUSINESS_ANALYST, RoleEnum.VIEWER
    ]:
        query = select(Role).where(Role.name == role_name)
        res = await session.execute(query)
        role = res.scalar_one_or_none()
        if not role:
            display_title = role_name.replace("_", " ").title()
            role = Role(name=role_name, display_name=display_title, is_system_role=True)
            session.add(role)
            await session.flush()
        role_objects[role_name] = role

        # Assign permissions to role
        patterns = ROLE_PERMISSION_MAP.get(role_name, [])
        role.permissions = []
        for code, perm in permission_objects.items():
            res, act = code.split(":")
            if "*:*" in patterns or f"{res}:*" in patterns or f"{res}:{act}" in patterns:
                role.permissions.append(perm)

    # 3. Create default Superuser if not present
    query = select(User).where(User.email == settings.FIRST_SUPERUSER_EMAIL)
    res = await session.execute(query)
    superuser = res.scalar_one_or_none()
    if not superuser:
        superuser = User(
            email=settings.FIRST_SUPERUSER_EMAIL,
            hashed_password=get_password_hash(settings.FIRST_SUPERUSER_PASSWORD),
            first_name=settings.FIRST_SUPERUSER_FIRSTNAME,
            last_name=settings.FIRST_SUPERUSER_LASTNAME,
            is_active=True,
            is_superuser=True,
            is_verified=True,
        )
        superuser.roles.append(role_objects[RoleEnum.SUPER_ADMIN])
        session.add(superuser)
        await session.flush()
        logger.info(f"Superuser '{settings.FIRST_SUPERUSER_EMAIL}' created.")

    # 4. Create default Organization if not present
    org_slug = "modelforge-enterprise"
    query = select(Organization).where(Organization.slug == org_slug)
    res = await session.execute(query)
    org = res.scalar_one_or_none()
    if not org:
        org = Organization(
            name=settings.FIRST_ORGANIZATION_NAME,
            slug=org_slug,
            description="Default enterprise organization for ModelForge AI.",
            plan="enterprise",
            max_projects=500,
            max_storage_gb=5000,
        )
        session.add(org)
        await session.flush()

        # Add superuser as organization owner
        membership = OrganizationMember(
            organization_id=org.id,
            user_id=superuser.id,
            role_id=role_objects[RoleEnum.ORG_ADMIN].id,
            is_owner=True,
        )
        session.add(membership)
        await session.flush()
        logger.info(f"Default organization '{org.name}' initialized.")

    await session.commit()
    logger.info("Database bootstrap completed successfully.")
