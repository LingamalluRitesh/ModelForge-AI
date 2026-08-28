"""
ModelForge AI - ML Engine: Causal Inference & Treatment Effect Estimation
Implements Double Machine Learning (DML), Propensity Score Matching (PSM),
Inverse Probability of Treatment Weighting (IPTW), Meta-Learners (S-Learner, T-Learner, X-Learner),
and Do-Calculus Directed Acyclic Graph (DAG) validation to quantify true causal effects.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from app.core.exceptions import MLModelExecutionException
from app.core.logging import logger


class PropensityScoreMatcher:
    """Estimates Average Treatment Effect on the Treated (ATT) via nearest-neighbor propensity matching."""

    def __init__(self, caliper: float = 0.05, replacement: bool = True):
        self.caliper = caliper
        self.replacement = replacement
        self.propensity_model = LogisticRegression(max_iter=500, random_state=42)
        self.is_fitted = False

    def fit_estimate(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Estimate treatment propensity scores $e(X) = P(T=1|X)$ and match treated units with control units.
        """
        X = np.asarray(X, dtype=np.float64)
        t = np.asarray(treatment, dtype=int)
        y = np.asarray(outcome, dtype=np.float64)

        self.propensity_model.fit(X, t)
        propensity_scores = self.propensity_model.predict_proba(X)[:, 1]
        self.is_fitted = True

        treated_idx = np.where(t == 1)[0]
        control_idx = np.where(t == 0)[0]

        treated_ps = propensity_scores[treated_idx]
        control_ps = propensity_scores[control_idx]

        matched_control_outcomes = []
        valid_treated_outcomes = []

        # Nearest neighbor matching on propensity score within caliper
        for i, ps_t in enumerate(treated_ps):
            distances = np.abs(control_ps - ps_t)
            min_dist_idx = np.argmin(distances)
            if distances[min_dist_idx] <= self.caliper:
                matched_control_outcomes.append(y[control_idx[min_dist_idx]])
                valid_treated_outcomes.append(y[treated_idx[i]])

        if not valid_treated_outcomes:
            att = float(np.mean(y[t == 1]) - np.mean(y[t == 0]))
            se = float(np.sqrt(np.var(y[t == 1]) / len(treated_idx) + np.var(y[t == 0]) / len(control_idx)))
        else:
            att = float(np.mean(valid_treated_outcomes) - np.mean(matched_control_outcomes))
            diffs = np.array(valid_treated_outcomes) - np.array(matched_control_outcomes)
            se = float(np.std(diffs) / np.sqrt(len(diffs))) if len(diffs) > 1 else 0.1

        z_score = att / max(1e-6, se)
        p_val = float(2 * (1 - norm.cdf(abs(z_score))))

        return {
            "method": "Propensity Score Matching (PSM)",
            "average_treatment_effect": round(att, 4),
            "standard_error": round(se, 4),
            "z_score": round(z_score, 4),
            "p_value": round(p_val, 4),
            "confidence_interval_95": [round(att - 1.96 * se, 4), round(att + 1.96 * se, 4)],
            "matched_pairs_count": len(valid_treated_outcomes),
            "mean_propensity_treated": round(float(np.mean(treated_ps)), 4),
            "mean_propensity_control": round(float(np.mean(control_ps)), 4),
        }


class DoubleMachineLearning:
    """
    Chernozhukov et al. Double/Debiased Machine Learning (DML) for causal parameter $\theta$.
    Residualizes outcome $Y$ and treatment $T$ using cross-fitted non-parametric nuisance models:
    $\tilde{Y} = Y - \hat{g}(X)$, $\tilde{T} = T - \hat{m}(X)$, $\tilde{Y} = \theta \tilde{T} + \epsilon$.
    """

    def __init__(self, n_splits: int = 3, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state

    def estimate_ate(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
    ) -> Dict[str, Any]:
        X = np.asarray(X, dtype=np.float64)
        T = np.asarray(treatment, dtype=np.float64)
        Y = np.asarray(outcome, dtype=np.float64)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        Y_res = np.zeros_like(Y)
        T_res = np.zeros_like(T)

        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            Y_tr, Y_val = Y[train_idx], Y[val_idx]
            T_tr, T_val = T[train_idx], T[val_idx]

            # Model 1: Outcome regression g(X)
            g_model = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=self.random_state)
            g_model.fit(X_tr, Y_tr)
            Y_res[val_idx] = Y_val - g_model.predict(X_val)

            # Model 2: Treatment propensity m(X)
            m_model = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=self.random_state)
            m_model.fit(X_tr, T_tr)
            T_res[val_idx] = T_val - m_model.predict(X_val)

        # Final Orthogonal Stage: Regress Y_res on T_res
        theta = float(np.sum(T_res * Y_res) / max(1e-6, np.sum(T_res ** 2)))
        
        # Influence function variance estimation
        n = len(Y)
        psi = (Y_res - theta * T_res) * T_res
        j_score = np.mean(T_res ** 2)
        variance = np.mean(psi ** 2) / max(1e-6, j_score ** 2)
        se = float(np.sqrt(variance / n))
        z_score = theta / max(1e-6, se)
        p_val = float(2 * (1 - norm.cdf(abs(z_score))))

        return {
            "method": "Double Machine Learning (DML)",
            "average_treatment_effect": round(theta, 4),
            "standard_error": round(se, 4),
            "z_score": round(z_score, 4),
            "p_value": round(p_val, 4),
            "confidence_interval_95": [round(theta - 1.96 * se, 4), round(theta + 1.96 * se, 4)],
            "sample_size": n,
        }


class MetaLearners:
    """Conditional Average Treatment Effect (CATE) estimators: S-Learner, T-Learner, and X-Learner."""

    @staticmethod
    def s_learner(
        X: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Single-model meta-learner including treatment as a standard feature."""
        X_aug = np.column_stack([X, treatment])
        model = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42)
        model.fit(X_aug, outcome)

        X_treated = np.column_stack([X, np.ones(len(X))])
        X_control = np.column_stack([X, np.zeros(len(X))])

        cate = model.predict(X_treated) - model.predict(X_control)
        ate = float(np.mean(cate))
        return cate, ate

    @staticmethod
    def t_learner(
        X: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Two-model meta-learner training separate models for treated and control groups."""
        mask_t = (treatment == 1)
        mask_c = (treatment == 0)

        m_treated = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42)
        m_treated.fit(X[mask_t], outcome[mask_t])

        m_control = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42)
        m_control.fit(X[mask_c], outcome[mask_c])

        cate = m_treated.predict(X) - m_control.predict(X)
        ate = float(np.mean(cate))
        return cate, ate
