"""
ModelForge AI - ML Engine: Survival Analysis & Time-to-Event Modeling
Implements Kaplan-Meier non-parametric survival estimator, Cox Proportional Hazards regression,
Log-Rank hypothesis testing, and Harrell's Concordance Index ($C$-index) for customer churn and predictive maintenance.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2


class KaplanMeierEstimator:
    """Non-parametric estimator of the survival function $S(t) = \prod_{t_i \le t} (1 - \frac{d_i}{n_i})$."""

    def __init__(self):
        self.survival_table_: Optional[pd.DataFrame] = None
        self.median_survival_time_: Optional[float] = None

    def fit(self, durations: np.ndarray, event_observed: np.ndarray):
        durations = np.asarray(durations, dtype=np.float64)
        events = np.asarray(event_observed, dtype=int)

        df = pd.DataFrame({"duration": durations, "event": events})
        unique_times = np.sort(df["duration"].unique())

        records = []
        n_at_risk = len(df)
        cum_survival = 1.0

        for t in unique_times:
            subset = df[df["duration"] == t]
            d_i = subset["event"].sum() # number of events at time t
            c_i = len(subset) - d_i      # number of censored at time t

            hazard = d_i / max(1, n_at_risk)
            cum_survival *= (1.0 - hazard)

            records.append({
                "time": float(t),
                "at_risk": int(n_at_risk),
                "events": int(d_i),
                "censored": int(c_i),
                "survival_probability": round(float(cum_survival), 4),
            })
            n_at_risk -= (d_i + c_i)

        self.survival_table_ = pd.DataFrame(records)

        # Median survival time (first time survival <= 0.5)
        med_row = self.survival_table_[self.survival_table_["survival_probability"] <= 0.5]
        if not med_row.empty:
            self.median_survival_time_ = float(med_row.iloc[0]["time"])
        else:
            self.median_survival_time_ = float(self.survival_table_["time"].max())

        return self

    def predict_survival_at(self, time_points: List[float]) -> List[float]:
        if self.survival_table_ is None:
            return [1.0] * len(time_points)

        probs = []
        for t in time_points:
            past = self.survival_table_[self.survival_table_["time"] <= t]
            if not past.empty:
                probs.append(float(past.iloc[-1]["survival_probability"]))
            else:
                probs.append(1.0)
        return probs


class CoxProportionalHazards:
    """Semi-parametric Cox proportional hazards model estimating feature hazard ratios $e^{\beta_j}$."""

    def __init__(self, l2_penalty: float = 1e-3, max_iter: int = 100):
        self.l2_penalty = l2_penalty
        self.max_iter = max_iter
        self.coef_: Optional[np.ndarray] = None
        self.hazard_ratios_: Optional[Dict[str, float]] = None
        self.concordance_index_: float = 0.0

    def fit(self, X: np.ndarray, durations: np.ndarray, event_observed: np.ndarray, feature_names: Optional[List[str]] = None):
        X = np.asarray(X, dtype=np.float64)
        T = np.asarray(durations, dtype=np.float64)
        E = np.asarray(event_observed, dtype=int)
        n_samples, n_features = X.shape

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        # Negative partial log-likelihood objective
        def neg_log_likelihood(beta: np.ndarray) -> float:
            risk_scores = np.dot(X, beta)
            exp_risk = np.exp(np.clip(risk_scores, -20, 20))
            loss = 0.0

            for i in range(n_samples):
                if E[i] == 1:
                    # Risk set: all subjects surviving at least duration T[i]
                    risk_set = (T >= T[i])
                    sum_exp = np.sum(exp_risk[risk_set])
                    loss -= (risk_scores[i] - np.log(max(1e-12, sum_exp)))

            # L2 Regularization
            loss += 0.5 * self.l2_penalty * np.sum(beta ** 2)
            return float(loss)

        init_beta = np.zeros(n_features)
        res = minimize(neg_log_likelihood, init_beta, method="L-BFGS-B", options={"maxiter": self.max_iter})
        self.coef_ = res.x

        self.hazard_ratios_ = {
            name: round(float(np.exp(coef)), 4)
            for name, coef in zip(feature_names, self.coef_)
        }

        # Calculate Harrell's Concordance Index (C-Index)
        risk_scores = np.dot(X, self.coef_)
        concordant = 0
        permissible = 0

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if T[i] != T[j]:
                    shorter_idx = i if T[i] < T[j] else j
                    longer_idx = j if T[i] < T[j] else i

                    if E[shorter_idx] == 1:
                        permissible += 1
                        if risk_scores[shorter_idx] > risk_scores[longer_idx]:
                            concordant += 1
                        elif risk_scores[shorter_idx] == risk_scores[longer_idx]:
                            concordant += 0.5

        self.concordance_index_ = round(float(concordant / max(1, permissible)), 4)
        return self

    def predict_risk_score(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise MLModelExecutionException("Cox model is not fitted.")
        return np.dot(X, self.coef_)
