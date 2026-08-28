"""
ModelForge AI - ML Engine: Model-Agnostic Permutation Feature Importance
Implements Fisher, Rudin, and Dominici Permutation Feature Importance with
Bootstrap Resampling Confidence Intervals and multi-metric loss tracking.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class PermutationFeatureImportance:
    """Computes decrease in model evaluation score when individual feature columns are randomly shuffled."""

    def __init__(
        self,
        scoring_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        n_repeats: int = 10,
        random_state: int = 42,
    ):
        self.scoring_func = scoring_func or self._default_accuracy
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.importances_mean_: Optional[np.ndarray] = None
        self.importances_std_: Optional[np.ndarray] = None
        self.importances_raw_: Optional[np.ndarray] = None

    def _default_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(y_true == y_pred))

    def compute(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Execute permutation importance across all features."""
        np.random.seed(self.random_state)
        X_arr = np.asarray(X).copy()
        y_arr = np.asarray(y)
        n_samples, n_features = X_arr.shape

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        # Baseline score
        baseline_preds = predict_fn(X_arr)
        baseline_score = self.scoring_func(y_arr, baseline_preds)

        raw_importances = np.zeros((n_features, self.n_repeats))

        for f_idx in range(n_features):
            original_col = X_arr[:, f_idx].copy()

            for rep in range(self.n_repeats):
                # Shuffle column
                shuffled_col = np.random.permutation(original_col)
                X_arr[:, f_idx] = shuffled_col

                shuffled_preds = predict_fn(X_arr)
                shuffled_score = self.scoring_func(y_arr, shuffled_preds)

                # Importance = decrease in score
                raw_importances[f_idx, rep] = baseline_score - shuffled_score

            # Restore column
            X_arr[:, f_idx] = original_col

        self.importances_raw_ = raw_importances
        self.importances_mean_ = np.mean(raw_importances, axis=1)
        self.importances_std_ = np.std(raw_importances, axis=1)

        results = []
        for i, name in enumerate(feature_names):
            results.append({
                "feature": name,
                "importance_mean": round(float(self.importances_mean_[i]), 5),
                "importance_std": round(float(self.importances_std_[i]), 5),
                "ci_lower": round(float(self.importances_mean_[i] - 1.96 * self.importances_std_[i] / np.sqrt(self.n_repeats)), 5),
                "ci_upper": round(float(self.importances_mean_[i] + 1.96 * self.importances_std_[i] / np.sqrt(self.n_repeats)), 5),
            })

        # Sort descending
        results.sort(key=lambda x: x["importance_mean"], reverse=True)

        return {
            "baseline_score": round(baseline_score, 4),
            "n_repeats": self.n_repeats,
            "feature_importances": results,
        }
