"""
ModelForge AI - Causal Inference & Policy Evaluation Domain Service
Executes propensity score matching, double machine learning, and uplift modeling for policy experiments.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.exceptions import EntityNotFoundException, ValidationException
from app.services.dataset_service import DatasetService
from ml_engine.algorithms.causal_inference import PropensityScoreMatcher, DoubleMachineLearning, MetaLearners


class CausalInferenceService:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.dataset_service = DatasetService(session)

    async def estimate_treatment_effects(
        self,
        dataset_version_id: str,
        treatment_column: str,
        outcome_column: str,
        covariate_columns: List[str],
        method: str = "dml",
    ) -> Dict[str, Any]:
        """Estimate causal treatment effect on target outcome."""
        df = await self.dataset_service.get_dataset_dataframe(dataset_version_id)
        if df.empty:
            raise EntityNotFoundException("Dataset version not found or empty.")

        if treatment_column not in df.columns or outcome_column not in df.columns:
            raise ValidationException(f"Columns {treatment_column} or {outcome_column} not found in dataset.")

        clean_df = df[[treatment_column, outcome_column] + covariate_columns].dropna()
        X = clean_df[covariate_columns].values
        T = clean_df[treatment_column].values
        Y = clean_df[outcome_column].values

        if method == "psm":
            matcher = PropensityScoreMatcher(caliper=0.05)
            return matcher.fit_estimate(X, T, Y)
        elif method == "dml":
            dml = DoubleMachineLearning(n_splits=3)
            return dml.estimate_ate(X, T, Y)
        elif method == "s_learner":
            cate, ate = MetaLearners.s_learner(X, T, Y)
            return {
                "method": "S-Learner (Single Model)",
                "average_treatment_effect": round(ate, 4),
                "cate_distribution": {
                    "min": round(float(np.min(cate)), 4),
                    "p25": round(float(np.percentile(cate, 25)), 4),
                    "median": round(float(np.median(cate)), 4),
                    "p75": round(float(np.percentile(cate, 75)), 4),
                    "max": round(float(np.max(cate)), 4),
                },
            }
        else:
            cate, ate = MetaLearners.t_learner(X, T, Y)
            return {
                "method": "T-Learner (Two-Model)",
                "average_treatment_effect": round(ate, 4),
                "cate_distribution": {
                    "median": round(float(np.median(cate)), 4),
                },
            }
