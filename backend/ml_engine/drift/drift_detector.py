"""
ModelForge AI - ML Engine: Data Drift & Concept Drift Detection
Implements Population Stability Index (PSI), Kolmogorov-Smirnov (KS-test),
Chi-Square Goodness-of-Fit, Wasserstein Distance, and Jensen-Shannon Divergence.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import distance


class DriftDetector:
    """Enterprise statistical drift detection engine."""

    @staticmethod
    def calculate_psi(baseline: np.ndarray, target: np.ndarray, num_buckets: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI) between baseline and target distributions.
        PSI < 0.10: No significant change
        0.10 <= PSI < 0.25: Moderate drift
        PSI >= 0.25: Significant drift
        """
        baseline = baseline[~np.isnan(baseline)]
        target = target[~np.isnan(target)]

        if len(baseline) == 0 or len(target) == 0:
            return 0.0

        # Create quantiles based on baseline
        percentiles = np.linspace(0, 100, num_buckets + 1)
        bin_edges = np.percentile(baseline, percentiles)
        bin_edges[0] -= 1e-5
        bin_edges[-1] += 1e-5

        # Handle identical bin edges (e.g. constant data)
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            return 0.0

        # Bucket counts
        base_counts, _ = np.histogram(baseline, bins=bin_edges)
        target_counts, _ = np.histogram(target, bins=bin_edges)

        # Normalize to fractions with smoothing to prevent log(0)
        eps = 1e-4
        base_pct = (base_counts + eps) / (len(baseline) + eps * len(base_counts))
        target_pct = (target_counts + eps) / (len(target) + eps * len(target_counts))

        # PSI formula: sum((Target_i - Base_i) * ln(Target_i / Base_i))
        psi_val = np.sum((target_pct - base_pct) * np.log(target_pct / base_pct))
        return float(max(0.0, psi_val))

    @staticmethod
    def calculate_ks_test(baseline: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
        """Kolmogorov-Smirnov two-sample test for continuous variables. Returns (statistic, p_value)."""
        baseline = baseline[~np.isnan(baseline)]
        target = target[~np.isnan(target)]
        if len(baseline) < 2 or len(target) < 2:
            return 0.0, 1.0
        res = stats.ks_2samp(baseline, target)
        return float(res.statistic), float(res.pvalue)

    @staticmethod
    def calculate_chi_square(baseline_categories: np.ndarray, target_categories: np.ndarray) -> Tuple[float, float]:
        """Chi-Square test for categorical feature distribution shift."""
        base_s = pd.Series(baseline_categories).dropna().astype(str)
        target_s = pd.Series(target_categories).dropna().astype(str)

        all_cats = list(set(base_s.unique()).union(set(target_s.unique())))
        base_counts = base_s.value_counts().reindex(all_cats, fill_value=0).values + 1
        target_counts = target_s.value_counts().reindex(all_cats, fill_value=0).values + 1

        res = stats.chisquare(f_obs=target_counts, f_exp=base_counts * (len(target_s) / len(base_s)))
        return float(res.statistic), float(res.pvalue)

    @staticmethod
    def calculate_wasserstein(baseline: np.ndarray, target: np.ndarray) -> float:
        """Wasserstein Distance (Earth Mover's Distance)."""
        baseline = baseline[~np.isnan(baseline)]
        target = target[~np.isnan(target)]
        if len(baseline) == 0 or len(target) == 0:
            return 0.0
        return float(stats.wasserstein_distance(baseline, target))

    @classmethod
    def evaluate_feature_drift(
        cls,
        baseline_df: pd.DataFrame,
        target_df: pd.DataFrame,
        psi_threshold_warning: float = 0.10,
        psi_threshold_critical: float = 0.25,
    ) -> Dict[str, Any]:
        """Run multi-metric statistical drift evaluation across all common columns."""
        common_cols = [c for c in baseline_df.columns if c in target_df.columns]
        feature_metrics = {}
        drifted_count = 0
        total_psi = 0.0

        for col in common_cols:
            is_num = pd.api.types.is_numeric_dtype(baseline_df[col]) and pd.api.types.is_numeric_dtype(target_df[col])

            if is_num:
                base_arr = baseline_df[col].dropna().values.astype(float)
                targ_arr = target_df[col].dropna().values.astype(float)

                psi = cls.calculate_psi(base_arr, targ_arr)
                ks_stat, ks_pval = cls.calculate_ks_test(base_arr, targ_arr)
                wass = cls.calculate_wasserstein(base_arr, targ_arr)

                has_drifted = psi >= psi_threshold_warning or ks_pval < 0.01
                if psi >= psi_threshold_critical:
                    severity = "critical"
                elif psi >= psi_threshold_warning:
                    severity = "warning"
                else:
                    severity = "none"

                if has_drifted:
                    drifted_count += 1
                total_psi += psi

                feature_metrics[col] = {
                    "feature_name": col,
                    "drift_score": round(psi, 4),
                    "algorithm_used": "psi_and_ks_test",
                    "p_value": round(ks_pval, 6),
                    "ks_statistic": round(ks_stat, 4),
                    "wasserstein_distance": round(wass, 4),
                    "has_drifted": has_drifted,
                    "severity": severity,
                }
            else:
                base_cats = baseline_df[col].dropna().astype(str).values
                targ_cats = target_df[col].dropna().astype(str).values

                chi_stat, chi_pval = cls.calculate_chi_square(base_cats, targ_cats)
                has_drifted = chi_pval < 0.05
                severity = "critical" if chi_pval < 0.001 else ("warning" if has_drifted else "none")

                if has_drifted:
                    drifted_count += 1

                feature_metrics[col] = {
                    "feature_name": col,
                    "drift_score": round(1.0 - chi_pval, 4),
                    "algorithm_used": "chi_square",
                    "p_value": round(chi_pval, 6),
                    "has_drifted": has_drifted,
                    "severity": severity,
                }

        total_cols = len(common_cols)
        avg_drift = (total_psi / max(1, total_cols))
        overall_severity = "critical" if avg_drift >= psi_threshold_critical or (drifted_count / max(1, total_cols)) > 0.4 else (
            "warning" if drifted_count > 0 else "low"
        )

        recommendations = []
        if overall_severity in ("warning", "critical"):
            recommendations.append(f"Significant drift detected in {drifted_count} of {total_cols} features.")
            recommendations.append("Trigger automated retraining pipeline with the newly ingested distribution.")

        return {
            "overall_drift_score": round(min(1.0, avg_drift), 4),
            "severity": overall_severity,
            "drifted_features_count": drifted_count,
            "total_features_count": total_cols,
            "feature_metrics": feature_metrics,
            "recommendations": recommendations,
        }
