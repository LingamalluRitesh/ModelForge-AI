"""
ModelForge AI - ML Engine: Model Evaluation & Metric Suite
Computes full classification metrics (Accuracy, Precision, Recall, F1, Balanced Acc,
ROC-AUC, PR-AUC, Confusion Matrix, ROC/PR curve vectors, and Calibration Curves).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)


class ClassificationEvaluator:
    """Enterprise classification evaluation engine."""

    @staticmethod
    def evaluate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        pos_label: Any = 1,
    ) -> Dict[str, Any]:
        """Compute full classification metrics dictionary."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        is_binary = len(unique_classes) <= 2

        acc = float(accuracy_score(y_true, y_pred))
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))

        if is_binary:
            prec = float(precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
            rec = float(recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
            f1 = float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
            prec_macro = prec
            rec_macro = rec
            f1_macro = f1
        else:
            prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
            rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
            f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
            prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
            rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
            f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
        cm_dict = {
            "matrix": cm.tolist(),
            "labels": [str(c) for c in unique_classes],
        }

        # Specificity / False Positive Rate for binary
        specificity = None
        fpr_score = None
        if is_binary and cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = float(tn / (tn + fp + 1e-12))
            fpr_score = float(fp / (tn + fp + 1e-12))

        # ROC-AUC & PR-AUC & Curve vectors
        roc_auc = None
        pr_auc = None
        roc_curve_data = None
        pr_curve_data = None

        if y_prob is not None:
            try:
                if is_binary:
                    # Select probabilities for positive class
                    if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                        prob_pos = y_prob[:, 1]
                    else:
                        prob_pos = y_prob.ravel()

                    # Convert y_true to binary (0, 1)
                    y_true_binary = (y_true == pos_label).astype(int)
                    roc_auc = float(roc_auc_score(y_true_binary, prob_pos))
                    pr_auc = float(average_precision_score(y_true_binary, prob_pos))

                    # ROC Curve coordinates (downsampled to max 50 points for efficient UI rendering)
                    fpr_pts, tpr_pts, _ = roc_curve(y_true_binary, prob_pos)
                    indices = np.linspace(0, len(fpr_pts) - 1, min(50, len(fpr_pts)), dtype=int)
                    roc_curve_data = [
                        {"fpr": float(fpr_pts[i]), "tpr": float(tpr_pts[i])}
                        for i in indices
                    ]

                    # PR Curve coordinates
                    pre_pts, rec_pts, _ = precision_recall_curve(y_true_binary, prob_pos)
                    pr_indices = np.linspace(0, len(pre_pts) - 1, min(50, len(pre_pts)), dtype=int)
                    pr_curve_data = [
                        {"recall": float(rec_pts[i]), "precision": float(pre_pts[i])}
                        for i in pr_indices
                    ]
                else:
                    if y_prob.ndim == 2 and y_prob.shape[1] == len(unique_classes):
                        roc_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
            except Exception:
                pass

        return {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "precision_macro": prec_macro,
            "recall_macro": rec_macro,
            "f1_macro": f1_macro,
            "specificity": specificity,
            "false_positive_rate": fpr_score,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "confusion_matrix": cm_dict,
            "roc_curve": roc_curve_data,
            "pr_curve": pr_curve_data,
        }


class RegressionEvaluator:
    """Enterprise regression evaluation engine."""

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray, n_features: Optional[int] = None) -> Dict[str, Any]:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        mae = float(mean_absolute_error(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_true, y_pred))

        # Adjusted R-squared
        n = len(y_true)
        p = n_features or 1
        adj_r2 = float(1 - (1 - r2) * (n - 1) / max(1, n - p - 1)) if n > (p + 1) else r2

        # Mean Absolute Percentage Error (avoiding divide-by-zero)
        denom = np.where(np.abs(y_true) < 1e-6, 1e-6, y_true)
        mape = float(np.mean(np.abs((y_true - y_pred) / denom))) * 100.0
        max_error = float(np.max(np.abs(y_true - y_pred)))

        # Residuals analysis
        residuals = (y_true - y_pred).tolist()
        residual_sample = residuals[:100] # for quick distribution plot

        return {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "adjusted_r2": adj_r2,
            "mape": mape,
            "max_error": max_error,
            "residual_sample": residual_sample,
        }
