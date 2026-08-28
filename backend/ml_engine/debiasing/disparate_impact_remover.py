"""
ModelForge AI - ML Engine: Disparate Impact Remover (Distribution Repair)
Edits continuous feature values to remove correlations with protected attributes
by aligning group-specific marginal distributions using Earth Mover's Distance / Quantile Mapping.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class DisparateImpactRemover:
    """Pre-processing feature repair algorithm aligning subgroup feature quantiles."""

    def __init__(self, repair_level: float = 1.0):
        """
        repair_level: float between 0.0 (no repair) and 1.0 (complete distribution alignment).
        """
        self.repair_level = np.clip(repair_level, 0.0, 1.0)
        self.medians_: Dict[int, float] = {}

    def fit_transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        sensitive_attribute: np.ndarray,
        feature_indices: Optional[List[int]] = None,
    ) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X_arr = X.values.copy().astype(np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64).copy()

        s_arr = np.asarray(sensitive_attribute)
        unique_groups = np.unique(s_arr)

        if len(unique_groups) < 2 or self.repair_level == 0.0:
            return X_arr

        if feature_indices is None:
            feature_indices = list(range(X_arr.shape[1]))

        for col_idx in feature_indices:
            col_data = X_arr[:, col_idx]
            # Estimate overall reference quantiles
            sorted_all = np.sort(col_data)

            # Repair group-by-group via quantile alignment
            for group in unique_groups:
                mask = (s_arr == group)
                if not np.any(mask):
                    continue

                group_vals = col_data[mask]
                # Compute empirical CDF percentiles for this group
                ranks = np.argsort(np.argsort(group_vals)) / max(1, len(group_vals) - 1)

                # Map percentiles to overall median reference distribution
                target_indices = np.clip((ranks * (len(sorted_all) - 1)).astype(int), 0, len(sorted_all) - 1)
                repaired_vals = sorted_all[target_indices]

                # Interpolate between original and fully repaired based on repair_level
                X_arr[mask, col_idx] = (1.0 - self.repair_level) * group_vals + self.repair_level * repaired_vals

        return X_arr
