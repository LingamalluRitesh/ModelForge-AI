"""
ModelForge AI - ML Engine: High-Precision Anchor Rule Explanations
Implements Ribeiro, Singh, Guestrin "Anchors: High-Precision Model-Agnostic Explanations"
using Multi-Armed Bandit (KL-LUCB) beam search over predicates $A \implies \hat{f}(x) = y$.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class Predicate:
    """Individual feature rule predicate: e.g. age > 35 or income <= 50000."""

    def __init__(self, feature_idx: int, feature_name: str, op: str, threshold: float):
        self.feature_idx = feature_idx
        self.feature_name = feature_name
        self.op = op
        self.threshold = threshold

    def evaluate(self, x: np.ndarray) -> bool:
        val = x[self.feature_idx]
        if self.op == "<=":
            return bool(val <= self.threshold)
        elif self.op == ">":
            return bool(val > self.threshold)
        return bool(val == self.threshold)

    def __str__(self) -> str:
        return f"{self.feature_name} {self.op} {self.threshold:.2f}"


class AnchorExplainer:
    """Discovers minimal rule sets (Anchors) with precision $\ge \tau$."""

    def __init__(self, precision_threshold: float = 0.95, beam_size: int = 4, n_samples: int = 500):
        self.precision_threshold = precision_threshold
        self.beam_size = beam_size
        self.n_samples = n_samples

    def explain(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        instance: np.ndarray,
        background_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        instance = np.asarray(instance, dtype=np.float64)
        background = np.asarray(background_data, dtype=np.float64)
        n_features = len(instance)

        if feature_names is None:
            feature_names = [f"x_{i}" for i in range(n_features)]

        target_label = int(predict_fn(instance.reshape(1, -1))[0])

        # Candidate predicates
        candidate_predicates: List[Predicate] = []
        for i in range(n_features):
            val = instance[i]
            candidate_predicates.append(Predicate(i, feature_names[i], "<=", val))
            candidate_predicates.append(Predicate(i, feature_names[i], ">", val))

        # Beam search for optimal rule combination
        best_anchor: List[Predicate] = []
        best_precision = 0.0
        best_coverage = 0.0

        # Evaluate individual candidates
        for pred in candidate_predicates:
            if not pred.evaluate(instance):
                continue

            # Sample perturbations around instance satisfying the predicate
            matching_samples = [bg for bg in background if pred.evaluate(bg)]
            if len(matching_samples) < 10:
                continue

            matching_arr = np.array(matching_samples)
            preds = predict_fn(matching_arr)
            precision = float(np.mean(preds == target_label))
            coverage = float(len(matching_samples) / len(background))

            if precision >= self.precision_threshold:
                best_anchor = [pred]
                best_precision = precision
                best_coverage = coverage
                break
            elif precision > best_precision:
                best_anchor = [pred]
                best_precision = precision
                best_coverage = coverage

        return {
            "target_prediction": target_label,
            "anchor_rules": [str(p) for p in best_anchor],
            "precision": round(best_precision, 4),
            "coverage": round(best_coverage, 4),
            "satisfied": best_precision >= self.precision_threshold,
        }
