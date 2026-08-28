"""
ModelForge AI - ML Engine: Spatial Kriging & Geostatistics
Implements Empirical Semivariogram fitting (Spherical, Exponential, Gaussian models)
and Ordinary Kriging optimal spatial interpolation for geographic & spatial sensor data.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.optimize import curve_fit


class VariogramModels:
    @staticmethod
    def spherical(h: np.ndarray, nugget: float, sill: float, range_val: float) -> np.ndarray:
        gamma = np.where(
            h <= range_val,
            nugget + sill * (1.5 * (h / max(1e-6, range_val)) - 0.5 * (h / max(1e-6, range_val)) ** 3),
            nugget + sill,
        )
        return np.where(h == 0, 0, gamma)

    @staticmethod
    def exponential(h: np.ndarray, nugget: float, sill: float, range_val: float) -> np.ndarray:
        gamma = nugget + sill * (1.0 - np.exp(-3.0 * h / max(1e-6, range_val)))
        return np.where(h == 0, 0, gamma)

    @staticmethod
    def gaussian(h: np.ndarray, nugget: float, sill: float, range_val: float) -> np.ndarray:
        gamma = nugget + sill * (1.0 - np.exp(-3.0 * (h / max(1e-6, range_val)) ** 2))
        return np.where(h == 0, 0, gamma)


class OrdinaryKriging:
    """Ordinary Kriging spatial interpolation engine."""

    def __init__(self, model_type: str = "spherical", n_lags: int = 15):
        self.model_type = model_type
        self.n_lags = n_lags
        self.coords_: Optional[np.ndarray] = None
        self.values_: Optional[np.ndarray] = None
        self.nugget_: float = 0.0
        self.sill_: float = 1.0
        self.range_: float = 1.0

    def fit(self, coordinates: np.ndarray, values: np.ndarray):
        """Fit empirical semivariogram from coordinates $(N, 2)$ and target values."""
        self.coords_ = np.asarray(coordinates, dtype=np.float64)
        self.values_ = np.asarray(values, dtype=np.float64)
        n = len(self.coords_)

        # Calculate pairwise distances and squared value differences
        dists = pdist(self.coords_)
        val_diffs = 0.5 * (pdist(self.values_[:, np.newaxis]) ** 2)

        max_dist = np.max(dists) / 2.0
        bins = np.linspace(0, max_dist, self.n_lags + 1)
        bin_indices = np.digitize(dists, bins) - 1

        lag_centers = []
        semivariances = []

        for b in range(self.n_lags):
            mask = bin_indices == b
            if np.any(mask):
                lag_centers.append(float(np.mean(dists[mask])))
                semivariances.append(float(np.mean(val_diffs[mask])))

        lag_centers = np.array(lag_centers)
        semivariances = np.array(semivariances)

        # Fit variogram model
        model_func = getattr(VariogramModels, self.model_type, VariogramModels.spherical)
        init_sill = float(np.var(self.values_))
        init_range = float(np.max(lag_centers)) * 0.5

        try:
            popt, _ = curve_fit(
                model_func,
                lag_centers,
                semivariances,
                p0=[0.0, init_sill, init_range],
                bounds=([0.0, 0.0, 1e-3], [np.inf, np.inf, np.inf]),
                maxfev=1000,
            )
            self.nugget_, self.sill_, self.range_ = popt
        except Exception:
            self.nugget_ = 0.0
            self.sill_ = init_sill
            self.range_ = init_range

        return self

    def predict(self, query_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict values and kriging variance at unseen locations."""
        queries = np.asarray(query_coords, dtype=np.float64)
        n_known = len(self.coords_)
        n_queries = len(queries)

        model_func = getattr(VariogramModels, self.model_type, VariogramModels.spherical)

        # Build known-to-known semivariance matrix with Lagrange multiplier
        D_known = squareform(pdist(self.coords_))
        Gamma = model_func(D_known, self.nugget_, self.sill_, self.range_)

        A = np.zeros((n_known + 1, n_known + 1))
        A[:n_known, :n_known] = Gamma
        A[:n_known, n_known] = 1.0
        A[n_known, :n_known] = 1.0
        A[n_known, n_known] = 0.0

        # Known-to-query semivariance
        D_query = cdist(self.coords_, queries)
        Gamma_query = model_func(D_query, self.nugget_, self.sill_, self.range_)

        b = np.ones((n_known + 1, n_queries))
        b[:n_known, :] = Gamma_query

        # Solve Kriging equations: A * weights = b
        weights = np.linalg.solve(A, b)

        w_known = weights[:n_known, :]
        lagrange = weights[n_known, :]

        # Target predictions: y* = sum(w_i * y_i)
        preds = np.dot(self.values_, w_known)

        # Kriging variance: sigma^2 = sum(w_i * gamma_i0) + mu
        var = np.sum(w_known * Gamma_query, axis=0) + lagrange
        var = np.maximum(0.0, var)

        return preds, np.sqrt(var)
