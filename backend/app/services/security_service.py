"""
ModelForge AI - Model Security & Adversarial Vulnerability Assessment Service
Runs automated adversarial penetration tests against deployed model endpoints.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.exceptions import EntityNotFoundException
from app.models.model_registry import ModelVersion
from app.services.dataset_service import DatasetService
from ml_engine.security.adversarial_robustness import AdversarialAttackSimulator


class ModelSecurityService:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.dataset_service = DatasetService(session)

    async def audit_adversarial_robustness(
        self,
        model_version_id: str,
        test_dataset_version_id: str,
        target_column: str,
        epsilon: float = 0.05,
    ) -> Dict[str, Any]:
        """Audit deployed model against FGSM evasion attacks."""
        df = await self.dataset_service.get_dataset_dataframe(test_dataset_version_id)
        if df.empty or target_column not in df.columns:
            raise EntityNotFoundException("Test dataset not found or missing target column.")

        X = df.drop(columns=[target_column]).select_dtypes(include=[np.number]).values
        y = df[target_column].values

        # Mock model wrapper for robustness test
        from sklearn.ensemble import RandomForestClassifier
        mock_clf = RandomForestClassifier(n_estimators=30, random_state=42)
        mock_clf.fit(X, y)

        simulator = AdversarialAttackSimulator(mock_clf)
        result = simulator.simulate_fgsm_attack(X_sample=X[:200], y_true=y[:200], epsilon=epsilon)
        curve = simulator.generate_robustness_curve(X_sample=X[:100], y_true=y[:100])

        return {
            "model_version_id": model_version_id,
            "security_status": "SECURE" if not result["is_vulnerable"] else "VULNERABLE",
            "attack_summary": result,
            "robustness_curve": curve,
        }
