"""
ModelForge AI - ML Engine: Partial Dependence Plots (PDP) & Individual Conditional Expectation (ICE)
Implements Friedman PDP marginal feature effect estimator and Goldstein ICE local curves.
$f_{PDP}(x_S) = \frac{1}{n} \sum_{i=1}^n \hat{f}(x_S, x_{C}^{(i)})$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class PartialDependenceEngine:
    """PDP and ICE marginal curve generator for arbitrary black-box estimators."""

    def __init__(self, grid_resolution: int = 50):
        self.grid_resolution = grid_resolution

    def compute_1d(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        X: Union[pd.DataFrame, np.ndarray],
        feature_idx: int,
        feature_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate PDP and ICE curves across grid values of a single target feature."""
        X_arr = np.asarray(X).copy()
        n_samples = X_arr.shape[0]

        col_vals = X_arr[:, feature_idx]
        grid_values = np.linspace(np.min(col_vals), np.max(col_vals), self.grid_resolution)

        # ICE matrix: shape (n_samples, grid_resolution)
        ice_curves = np.zeros((n_samples, self.grid_resolution))

        X_temp = X_arr.copy()
        for g_idx, val in enumerate(grid_values):
            X_temp[:, feature_idx] = val
            preds = predict_fn(X_temp)
            ice_curves[:, g_idx] = preds

        pdp_curve = np.mean(ice_curves, axis=0)
        pdp_std = np.std(ice_curves, axis=0)

        # Centered ICE (c-ICE)
        centered_ice = ice_curves - ice_curves[:, [0]]

        return {
            "feature_index": feature_idx,
            "feature_name": feature_name or f"feature_{feature_idx}",
            "grid_values": grid_values.tolist(),
            "pdp_mean": pdp_curve.tolist(),
            "pdp_std": pdp_std.tolist(),
            "ice_samples": ice_curves[: min(20, n_samples)].tolist(),  # Subsample 20 curves for rendering
            "centered_ice_samples": centered_ice[: min(20, n_samples)].tolist(),
        }
