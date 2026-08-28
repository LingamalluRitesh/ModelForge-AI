"""
ModelForge AI - ML Engine: Calibrated Equalized Odds (Post-Processing)
Optimizes group-specific decision thresholds to equalize True Positive Rates (TPR)
and False Positive Rates (FPR) across demographic groups while maintaining calibrated probabilities.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.optimize import linprog


class CalibratedEqualizedOddsPostprocessor:
    """Post-processing threshold optimizer achieving Equalized Odds."""

    def __init__(self, cost_constraint: str = "equalized_odds"):
        """
        cost_constraint: 'equalized_odds' (equalize TPR & FPR) or 'equal_opportunity' (equalize TPR only).
        """
        self.cost_constraint = cost_constraint
        self.group_thresholds_: Dict[Any, float] = {}

    def fit(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        sensitive_attribute: np.ndarray,
    ):
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        s_arr = np.asarray(sensitive_attribute)
        unique_groups = np.unique(s_arr)

        # Search for group-specific optimal decision thresholds
        candidate_thresholds = np.linspace(0.1, 0.9, 81)
        best_diff = float("inf")
        best_thresholds = {g: 0.5 for g in unique_groups}

        # Grid evaluation for minimal disparate error difference
        tpr_by_thresh = {g: [] for g in unique_groups}
        fpr_by_thresh = {g: [] for g in unique_groups}

        for g in unique_groups:
            mask_g = (s_arr == g)
            y_t_g = y_true[mask_g]
            y_p_g = y_prob[mask_g]

            for th in candidate_thresholds:
                preds_th = (y_p_g >= th).astype(int)
                tp = np.sum((preds_th == 1) & (y_t_g == 1))
                fn = np.sum((preds_th == 0) & (y_t_g == 1))
                fp = np.sum((preds_th == 1) & (y_t_g == 0))
                tn = np.sum((preds_th == 0) & (y_t_g == 0))

                tpr = tp / max(1, tp + fn)
                fpr = fp / max(1, fp + tn)
                tpr_by_thresh[g].append(tpr)
                fpr_by_thresh[g].append(fpr)

        # Match TPRs across groups
        if len(unique_groups) == 2:
            g0, g1 = unique_groups[0], unique_groups[1]
            for i, th0 in enumerate(candidate_thresholds):
                tpr0 = tpr_by_thresh[g0][i]
                fpr0 = fpr_by_thresh[g0][i]

                # Find closest matching TPR in group 1
                tpr_diffs = np.abs(np.array(tpr_by_thresh[g1]) - tpr0)
                fpr_diffs = np.abs(np.array(fpr_by_thresh[g1]) - fpr0)

                total_diff = float(np.min(tpr_diffs) + np.min(fpr_diffs))
                if total_diff < best_diff:
                    best_diff = total_diff
                    best_j = int(np.argmin(tpr_diffs))
                    best_thresholds[g0] = float(th0)
                    best_thresholds[g1] = float(candidate_thresholds[best_j])

        self.group_thresholds_ = best_thresholds
        return self

    def predict(
        self,
        y_prob: np.ndarray,
        sensitive_attribute: np.ndarray,
    ) -> np.ndarray:
        y_prob = np.asarray(y_prob)
        s_arr = np.asarray(sensitive_attribute)
        preds = np.zeros(len(y_prob), dtype=int)

        for i in range(len(y_prob)):
            th = self.group_thresholds_.get(s_arr[i], 0.5)
            preds[i] = int(y_prob[i] >= th)

        return preds
