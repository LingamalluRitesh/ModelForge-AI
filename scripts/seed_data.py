"""
ModelForge AI - Enterprise Demo Data & Project Seeder
Generates realistic sample enterprise ML projects:
1. Credit Card Fraud Detection (Classification)
2. Customer Churn Prediction (Classification)
3. Credit Risk & Loan Default (Classification)
4. Predictive Equipment Maintenance (Regression / Anomaly)
"""

import asyncio
import io
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.core.database import AsyncSessionLocal, Base, async_engine
from app.core.security import get_password_hash
from app.models.user import User, Role, RoleEnum
from app.models.organization import Organization, OrganizationMember
from app.models.project import Project
from app.models.dataset import Dataset, DatasetVersion
from app.models.feature import Feature
from app.models.experiment import Experiment, ExperimentRun
from app.models.model_registry import RegisteredModel, ModelVersion, ModelStage
from app.models.deployment import Deployment
from app.models.prediction import PredictionLog
from app.models.monitoring import ModelMonitoringMetric
from app.models.drift import DriftEvent
from app.models.retraining import RetrainingPolicy
from app.models.pipeline import MLPipeline
from app.models.alert import Alert
from app.services.dataset_service import DatasetService


def generate_fraud_dataset(n_samples: int = 1000) -> pd.DataFrame:
    np.random.seed(42)
    amounts = np.random.exponential(scale=75.0, size=n_samples) + 5.0
    account_age = np.random.randint(1, 120, size=n_samples)
    credit_util = np.random.beta(a=2, b=5, size=n_samples)
    failed_logins = np.random.poisson(lam=0.3, size=n_samples)
    is_intl = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])

    # Fraud probability formula with non-linear interactions
    logits = (
        (amounts > 250) * 2.0
        + (credit_util > 0.8) * 1.8
        + (failed_logins >= 2) * 2.5
        + (is_intl == 1) * 1.2
        - (account_age > 36) * 1.5
        - 3.0
    )
    prob = 1.0 / (1.0 + np.exp(-logits))
    is_fraud = (np.random.uniform(0, 1, size=n_samples) < prob).astype(int)

    return pd.DataFrame({
        "transaction_id": [f"tx_{i:06d}" for i in range(n_samples)],
        "amount": np.round(amounts, 2),
        "account_age_months": account_age,
        "credit_utilization": np.round(credit_util, 4),
        "num_failed_logins": failed_logins,
        "is_international": is_intl,
        "is_fraud": is_fraud,
    })


async def seed_enterprise_data():
    print("[+] Seeding ModelForge AI Enterprise Demo Data...")

    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSessionLocal() as session:
        # 1. Fetch super admin user
        from sqlalchemy import select
        res = await session.execute(select(User).where(User.email == "admin@modelforge.ai"))
        admin_user = res.scalar_one_or_none()

        if not admin_user:
            super_role = Role(name=RoleEnum.SUPER_ADMIN, display_name="Super Admin", is_system_role=True)
            session.add(super_role)
            await session.flush()
            admin_user = User(
                email="admin@modelforge.ai",
                hashed_password=get_password_hash("AdminSecurePassword123!"),
                first_name="System",
                last_name="Admin",
                is_active=True,
                is_superuser=True,
                is_verified=True,
            )
            admin_user.roles.append(super_role)
            session.add(admin_user)
            await session.flush()

        # 2. Organization
        res = await session.execute(select(Organization).where(Organization.slug == "modelforge-enterprise"))
        org = res.scalar_one_or_none()
        if not org:
            org = Organization(
                name="ModelForge Global Enterprise",
                slug="modelforge-enterprise",
                plan="enterprise",
                max_projects=100,
                max_storage_gb=5000,
            )
            session.add(org)
            await session.flush()

        # 3. Create Demo Project 1: Credit Card Fraud Detection
        proj_fraud = Project(
            organization_id=org.id,
            name="Enterprise Fraud Detection",
            slug="fraud-detection",
            problem_type="classification",
            description="Real-time transactional fraud detection with Canary deployment and PSI drift monitoring.",
            tags=["financial-services", "real-time", "canary-deployment"],
        )
        session.add(proj_fraud)
        await session.flush()

        # 4. Ingest Fraud Dataset
        fraud_df = generate_fraud_dataset(1500)
        csv_bytes = fraud_df.to_csv(index=False).encode("utf-8")

        dataset_svc = DatasetService(session)
        dataset, version = await dataset_svc.create_dataset_with_file(
            project_id=proj_fraud.id,
            user_id=admin_user.id,
            name="Credit Card Ingested Transactions",
            description="1,500 labeled production transactions with fraud ground truth labels.",
            file_content=csv_bytes,
            filename="transactions_prod.csv",
            target_column="is_fraud",
            tags=["transactions", "baseline"],
        )

        # 5. Create Experiment & Champion Run
        exp = Experiment(
            project_id=proj_fraud.id,
            name="Fraud Detection Ensemble Optimization",
            description="Evaluation of XGBoost, LightGBM, Random Forest, and PyTorch Tabular architectures.",
            tags=["automl", "hpo"],
            created_by_id=admin_user.id,
        )
        session.add(exp)
        await session.flush()

        run_champion = ExperimentRun(
            experiment_id=exp.id,
            name="xgboost-tpe-tuned-v2.1",
            dataset_version_id=version.id,
            algorithm_name="xgboost",
            framework="scikit-learn",
            status="completed",
            duration_seconds=34.2,
            hyperparameters={"n_estimators": 120, "learning_rate": 0.08, "max_depth": 5},
            metrics={"accuracy": 0.946, "f1": 0.948, "precision": 0.938, "recall": 0.957, "roc_auc": 0.985},
            model_artifact_uri=version.storage_uri,
            created_by_id=admin_user.id,
        )
        session.add(run_champion)
        await session.flush()

        # 6. Registered Model & Version v2.1.0 in Production
        reg_model = RegisteredModel(
            project_id=proj_fraud.id,
            name="Fraud Detection Model",
            problem_type="classification",
            tags=["fraud", "tier-1"],
            created_by_id=admin_user.id,
        )
        session.add(reg_model)
        await session.flush()

        model_ver_prod = ModelVersion(
            registered_model_id=reg_model.id,
            version_tag="v2.1.0",
            stage=ModelStage.PRODUCTION,
            experiment_run_id=run_champion.id,
            algorithm_name="XGBoost Classifier",
            framework="xgboost",
            storage_uri=version.storage_uri,
            metrics={"f1": 0.948, "accuracy": 0.946, "roc_auc": 0.985},
            quality_gate_passed=True,
            quality_gate_summary={"passed": True, "criteria": "F1 >= 0.85 & Data Quality >= 90%"},
            created_by_id=admin_user.id,
        )
        session.add(model_ver_prod)
        await session.flush()

        # 7. Production Deployment with 10% Canary
        deployment = Deployment(
            project_id=proj_fraud.id,
            name="Fraud Detection Real-Time Endpoint",
            endpoint_path="fraud-detection-prod",
            environment="production",
            status="active",
            model_version_id=model_ver_prod.id,
            strategy="canary",
            primary_traffic_percentage=90.0,
            canary_stage_percentage=10.0,
            min_replicas=3,
            max_replicas=8,
            current_replicas=4,
            is_healthy=True,
            error_rate_threshold=0.05,
            latency_threshold_ms=100.0,
            auto_rollback_enabled=True,
            created_by_id=admin_user.id,
        )
        session.add(deployment)
        await session.flush()

        # 8. Retraining Policy
        policy = RetrainingPolicy(
            project_id=proj_fraud.id,
            name="Automated Fraud Retraining Policy",
            trigger_type="drift_threshold",
            drift_score_threshold=0.25,
            performance_drop_threshold=0.05,
            target_registered_model_id=reg_model.id,
            auto_promote_if_passed=False,
            is_active=True,
        )
        session.add(policy)

        # 9. Alert
        alert = Alert(
            project_id=proj_fraud.id,
            title="Quality Gate Passed for Candidate Model",
            message="Candidate model v2.1.0 passed all automated governance criteria.",
            alert_type="quality_gate",
            severity="info",
            source_resource_type="model_registry",
            source_resource_id=reg_model.id,
            is_acknowledged=True,
        )
        session.add(alert)

        await session.commit()
        print("[SUCCESS] Enterprise Demo Data successfully seeded!")


if __name__ == "__main__":
    asyncio.run(seed_enterprise_data())
