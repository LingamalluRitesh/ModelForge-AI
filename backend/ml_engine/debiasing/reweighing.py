"""
ModelForge AI - ML Engine: Reweighing Fairness Preprocessor
Implements pre-processing instance weight adjustment to achieve Demographic Parity (Statistical Parity)
without altering the original feature values.
$W(s, y) = \frac{P(S=s) \times P(Y=y)}{P(S=s, Y=y)}$
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class ReweighingPreprocessor:
    """Pre-processing instance reweighing algorithm to neutralize statistical bias."""

    def __init__(self, favorable_label: Any = 1, privileged_group: Any = 1):
        self.favorable_label = favorable_label
        self.privileged_group = privileged_group
        self.weights_map_: Dict[Tuple[Any, Any], float] = {}

    def fit(self, sensitive_attributes: np.ndarray, labels: np.ndarray):
        s_arr = np.asarray(sensitive_attributes)
        y_arr = np.asarray(labels)
        n = len(s_arr)

        unique_s = np.unique(s_arr)
        unique_y = np.unique(y_arr)

        p_s = {s_val: np.mean(s_arr == s_val) for s_val in unique_s}
        p_y = {y_val: np.mean(y_arr == y_val) for y_val in unique_y}

        self.weights_map_ = {}
        for s_val in unique_s:
            for y_val in unique_y:
                mask_sy = (s_arr == s_val) & (y_arr == y_val)
                p_sy = np.mean(mask_sy)
                if p_sy > 0:
                    weight = (p_s[s_val] * p_y[y_val]) / p_sy
                else:
                    weight = 1.0
                self.weights_map_[(s_val, y_val)] = float(weight)

        return self

    def transform(self, sensitive_attributes: np.ndarray, labels: np.ndarray) -> np.ndarray:
        s_arr = np.asarray(sensitive_attributes)
        y_arr = np.asarray(labels)

        weights = np.zeros(len(s_arr), dtype=np.float32)
        for i in range(len(s_arr)):
            weights[i] = self.weights_map_.get((s_arr[i], y_arr[i]), 1.0)

        # Normalize weights to preserve total sample sum
        weights = weights * (len(weights) / np.sum(weights))
        return weights

    def fit_transform(self, sensitive_attributes: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return self.fit(sensitive_attributes, labels).transform(sensitive_attributes, labels)
