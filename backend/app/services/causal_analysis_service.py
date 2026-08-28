"""
ModelForge AI - Causal Inference & Uplift Service
Provides Double Machine Learning, Propensity Score Matching, and Heterogeneous Treatment Effect (CATE) analytics.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from ml_engine.causal.double_ml_suite import DoubleMLPLR


class CausalAnalysisService:
    @staticmethod
    def estimate_treatment_effect(
        records: List[Dict[str, Any]],
        treatment_column: str,
        outcome_column: str,
        confounders: List[str],
        method: str = "dml",
    ) -> Dict[str, Any]:
        """Estimate causal effect of treatment on outcome variable."""
        df = pd.DataFrame(records)

        X = df[confounders].values.astype(float)
        d = df[treatment_column].values.astype(float)
        y = df[outcome_column].values.astype(float)

        dml = DoubleMLPLR(n_folds=5).fit(X, d, y)

        return {
            "method": method,
            "treatment_column": treatment_column,
            "outcome_column": outcome_column,
            "confounders": confounders,
            "estimated_ate": round(dml.coef_, 4),
            "standard_error": round(dml.se_, 4),
            "t_statistic": round(dml.t_stat_, 3),
            "ci_lower": round(dml.ci_lower_, 4),
            "ci_upper": round(dml.ci_upper_, 4),
            "statistically_significant": abs(dml.t_stat_) > 1.96,
        }
