"""
ModelForge AI - ML Engine: Caruana Ensemble Selection & Hill-Climbing Blender
Implements post-hoc greedy ensemble selection with replacement (Rich Caruana algorithm)
optimizing any custom performance metric without overfitting through out-of-fold validation predictions.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score


class CaruanaEnsembleSelector:
    """Greedy ensemble selection with replacement across diverse candidate models."""

    def __init__(
        self,
        ensemble_size: int = 50,
        metric: str = "f1",
        problem_type: str = "classification",
    ):
        self.ensemble_size = ensemble_size
        self.metric = metric
        self.problem_type = problem_type

        self.selected_model_indices_: List[int] = []
        self.model_weights_: Dict[int, float] = {}
        self.best_ensemble_score_: float = 0.0

    def _score_predictions(self, y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
        if self.problem_type == "classification":
            if y_pred_prob.ndim == 2 and y_pred_prob.shape[1] == 2:
                # Binary
                y_pred_labels = (y_pred_prob[:, 1] >= 0.5).astype(int)
                if self.metric == "roc_auc":
                    return float(roc_auc_score(y_true, y_pred_prob[:, 1]))
                elif self.metric == "accuracy":
                    return float(accuracy_score(y_true, y_pred_labels))
                else:
                    return float(f1_score(y_true, y_pred_labels, zero_division=0))
            else:
                y_pred_labels = np.argmax(y_pred_prob, axis=1) if y_pred_prob.ndim > 1 else y_pred_prob
                return float(f1_score(y_true, y_pred_labels, average="weighted", zero_division=0))
        else:
            if self.metric == "r2":
                return float(r2_score(y_true, y_pred_prob))
            # Negative RMSE (higher is better)
            return -float(np.sqrt(mean_squared_error(y_true, y_pred_prob)))

    def fit(self, candidate_oof_predictions: List[np.ndarray], y_true: np.ndarray):
        """
        candidate_oof_predictions: List of $M$ numpy prediction arrays from $M$ candidate models.
        y_true: Ground truth target vector.
        """
        n_models = len(candidate_oof_predictions)
        if n_models == 0:
            return self

        # 1. Find the best individual starting model
        best_init_score = -float("inf")
        best_init_idx = 0

        for i, preds in enumerate(candidate_oof_predictions):
            score = self._score_predictions(y_true, preds)
            if score > best_init_score:
                best_init_score = score
                best_init_idx = i

        self.selected_model_indices_ = [best_init_idx]
        current_ensemble_sum = candidate_oof_predictions[best_init_idx].copy().astype(np.float64)

        # 2. Greedy hill-climbing with replacement
        for step in range(1, self.ensemble_size):
            best_step_score = -float("inf")
            best_step_idx = 0

            for i, preds in enumerate(candidate_oof_predictions):
                # Trial blend: (current_ensemble_sum + candidate_i) / (step + 1)
                trial_preds = (current_ensemble_sum + preds) / (step + 1)
                score = self._score_predictions(y_true, trial_preds)

                if score > best_step_score:
                    best_step_score = score
                    best_step_idx = i

            # Add winning model to ensemble
            self.selected_model_indices_.append(best_step_idx)
            current_ensemble_sum += candidate_oof_predictions[best_step_idx]

        self.best_ensemble_score_ = self._score_predictions(
            y_true, current_ensemble_sum / self.ensemble_size
        )

        # Compute normalized model weights
        from collections import Counter
        counts = Counter(self.selected_model_indices_)
        self.model_weights_ = {
            idx: count / self.ensemble_size for idx, count in counts.items()
        }

        return self

    def blend(self, candidate_test_predictions: List[np.ndarray]) -> np.ndarray:
        """Combine test predictions using optimal ensemble weights."""
        blend_sum = None
        for idx, weight in self.model_weights_.items():
            preds = candidate_test_predictions[idx] * weight
            if blend_sum is None:
                blend_sum = preds.copy()
            else:
                blend_sum += preds
        return blend_sum
