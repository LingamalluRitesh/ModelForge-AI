"""
ModelForge AI - ML Engine: Model Calibration & Uncertainty Estimation
Implements Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
Platt Scaling (Logistic Sigmoid), and Isotonic Regression Probability Calibrator.
$	ext{ECE} = \sum_{m=1}^M rac{|B_m|}{N} |	ext{acc}(B_m) - 	ext{conf}(B_m)|$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class CalibrationEvaluator:
    """Computes Expected Calibration Error (ECE) and reliability diagrams across probability bins."""
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def evaluate(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        y_true = np.asarray(y_true, dtype=int)
        y_prob = np.asarray(y_prob, dtype=float)
        N = len(y_true)

        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        mce = 0.0
        bin_records = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = float(np.mean(in_bin))

            if prop_in_bin > 0:
                accuracy_in_bin = float(np.mean(y_true[in_bin]))
                avg_confidence_in_bin = float(np.mean(y_prob[in_bin]))
                diff = abs(accuracy_in_bin - avg_confidence_in_bin)

                ece += diff * prop_in_bin
                mce = max(mce, diff)

                bin_records.append({
                    "bin_range": f"{bin_lower:.2f}-{bin_upper:.2f}",
                    "confidence": round(avg_confidence_in_bin, 4),
                    "accuracy": round(accuracy_in_bin, 4),
                    "sample_count": int(np.sum(in_bin)),
                })

        return {
            "expected_calibration_error": round(ece, 4),
            "maximum_calibration_error": round(mce, 4),
            "n_bins": self.n_bins,
            "reliability_curve": bin_records,
            "calibrated": ece < 0.05,
        }


class PlattScaler:
    """Post-hoc probability calibration via univariate logistic sigmoid fitting."""
    def __init__(self):
        self.A_: float = 1.0
        self.B_: float = 0.0

    def fit(self, logits: np.ndarray, y_true: np.ndarray, max_iter: int = 100, lr: float = 0.01):
        z = np.asarray(logits, dtype=float)
        y = np.asarray(y_true, dtype=float)
        N = len(y)

        A = 1.0
        B = 0.0

        for _ in range(max_iter):
            p = 1.0 / (1.0 + np.exp(-(A * z + B)))
            err = p - y
            grad_A = np.mean(err * z)
            grad_B = np.mean(err)
            A -= lr * grad_A
            B -= lr * grad_B

        self.A_ = A
        self.B_ = B
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        z = np.asarray(logits, dtype=float)
        return 1.0 / (1.0 + np.exp(-(self.A_ * z + self.B_)))
