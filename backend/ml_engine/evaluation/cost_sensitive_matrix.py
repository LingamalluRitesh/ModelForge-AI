"""
ModelForge AI - Evaluation: Cost-Sensitive Matrix & Optimal Decision Thresholding
Calculates expected financial loss using asymmetric False Positive (FP) and False Negative (FN) cost penalties.
$	ext{Total Cost} = C_{FP} \cdot FP + C_{FN} \cdot FN + C_{TP} \cdot TP + C_{TN} \cdot TN$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class CostSensitiveThresholdOptimizer:
    """Discovers optimal probability cut threshold $	au^*$ minimizing enterprise dollar loss."""
    def __init__(self, cost_fp: float = 10.0, cost_fn: float = 500.0, cost_tp: float = 0.0, cost_tn: float = 0.0):
        self.c_fp = cost_fp
        self.c_fn = cost_fn
        self.c_tp = cost_tp
        self.c_tn = cost_tn

    def optimize_threshold(self, y_true: np.ndarray, y_prob: np.ndarray, num_cuts: int = 100) -> Dict[str, Any]:
        y_true = np.asarray(y_true, dtype=int)
        y_prob = np.asarray(y_prob, dtype=float)

        thresholds = np.linspace(0.01, 0.99, num_cuts)
        best_thresh = 0.5
        min_cost = float("inf")
        curve = []

        for tau in thresholds:
            y_pred = (y_prob >= tau).astype(int)
            tp = int(np.sum((y_true == 1) & (y_pred == 1)))
            fp = int(np.sum((y_true == 0) & (y_pred == 1)))
            tn = int(np.sum((y_true == 0) & (y_pred == 0)))
            fn = int(np.sum((y_true == 1) & (y_pred == 0)))

            total_cost = self.c_fp * fp + self.c_fn * fn + self.c_tp * tp + self.c_tn * tn

            curve.append({
                "threshold": round(float(tau), 2),
                "total_cost": round(float(total_cost), 2),
                "false_positives": fp,
                "false_negatives": fn,
            })

            if total_cost < min_cost:
                min_cost = total_cost
                best_thresh = float(tau)

        return {
            "optimal_threshold": round(best_thresh, 3),
            "minimum_expected_cost_usd": round(min_cost, 2),
            "cost_matrix": {"C_FP": self.c_fp, "C_FN": self.c_fn},
            "threshold_cost_curve": curve,
        }
