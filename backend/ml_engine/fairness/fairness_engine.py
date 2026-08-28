"""
ModelForge AI - ML Engine: Model Fairness & Bias Audit
Calculates Demographic Parity, Disparate Impact (4/5ths rule),
Equal Opportunity Difference, and Equalized Odds across protected attributes.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class FairnessAuditEngine:
    """Enterprise AI algorithmic fairness and bias assessment engine."""

    @staticmethod
    def audit_model_fairness(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attribute_values: np.ndarray,
        favorable_label: Any = 1,
    ) -> Dict[str, Any]:
        """
        Evaluate fairness metrics across distinct subgroups of the sensitive attribute.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        groups = np.asarray(sensitive_attribute_values)

        unique_groups = np.unique(groups)
        if len(unique_groups) < 2:
            return {
                "demographic_parity_ratio": 1.0,
                "disparate_impact_ratio": 1.0,
                "equal_opportunity_difference": 0.0,
                "equalized_odds_difference": 0.0,
                "is_fair": True,
                "group_metrics": {},
                "recommendations": ["Sensitive attribute must have at least 2 distinct groups."],
            }

        group_metrics = {}
        selection_rates = {}
        tpr_rates = {}
        fpr_rates = {}

        for g in unique_groups:
            mask = groups == g
            y_t_g = y_true[mask]
            y_p_g = y_pred[mask]
            total_g = len(y_t_g)

            if total_g == 0:
                continue

            # Selection Rate: P(y_hat = favorable)
            pos_preds = np.sum(y_p_g == favorable_label)
            sel_rate = float(pos_preds / total_g)
            selection_rates[str(g)] = sel_rate

            # True Positive Rate (TPR): P(y_hat = favorable | y = favorable)
            actual_pos_mask = y_t_g == favorable_label
            actual_pos_count = np.sum(actual_pos_mask)
            tpr = float(np.sum(y_p_g[actual_pos_mask] == favorable_label) / actual_pos_count) if actual_pos_count > 0 else 1.0
            tpr_rates[str(g)] = tpr

            # False Positive Rate (FPR): P(y_hat = favorable | y != favorable)
            actual_neg_mask = y_t_g != favorable_label
            actual_neg_count = np.sum(actual_neg_mask)
            fpr = float(np.sum(y_p_g[actual_neg_mask] == favorable_label) / actual_neg_count) if actual_neg_count > 0 else 0.0
            fpr_rates[str(g)] = fpr

            group_metrics[str(g)] = {
                "sample_count": int(total_g),
                "selection_rate": sel_rate,
                "true_positive_rate": tpr,
                "false_positive_rate": fpr,
            }

        # Demographic Parity: min(selection_rate) / max(selection_rate)
        rates = list(selection_rates.values())
        min_rate, max_rate = min(rates), max(rates)
        disparate_impact = float(min_rate / (max_rate + 1e-12)) if max_rate > 0 else 1.0

        # Equal Opportunity Difference: max(|TPR_a - TPR_b|)
        tprs = list(tpr_rates.values())
        eq_opp_diff = float(max(tprs) - min(tprs))

        # Equalized Odds Difference: max(|TPR_a - TPR_b|, |FPR_a - FPR_b|)
        fprs = list(fpr_rates.values())
        fpr_diff = float(max(fprs) - min(fprs))
        eq_odds_diff = float(max(eq_opp_diff, fpr_diff))

        # Check 4/5ths (0.80) rule for disparate impact and < 0.10 for opportunity difference
        is_fair = (disparate_impact >= 0.80) and (eq_opp_diff <= 0.15)

        recommendations = []
        if disparate_impact < 0.80:
            recommendations.append(
                f"Disparate impact ratio is {disparate_impact:.2f} (< 0.80). Consider re-weighting training samples or applying threshold tuning."
            )
        if eq_opp_diff > 0.15:
            recommendations.append(
                f"Equal opportunity difference is {eq_opp_diff:.2f} (> 0.15). Model exhibits disparate false negative rates across groups."
            )
        if is_fair:
            recommendations.append("Model satisfies standard 4/5ths fairness guidelines for the evaluated attribute.")

        return {
            "demographic_parity_ratio": disparate_impact,
            "disparate_impact_ratio": disparate_impact,
            "equal_opportunity_difference": eq_opp_diff,
            "equalized_odds_difference": eq_odds_diff,
            "is_fair": is_fair,
            "group_metrics": group_metrics,
            "recommendations": recommendations,
        }
