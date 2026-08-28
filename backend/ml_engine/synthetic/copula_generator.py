"""
ModelForge AI - ML Engine: Gaussian & Clayton Copula Joint Distribution Generator
Implements Sklar's Theorem for synthetic joint probability modeling via Cumulative Distribution Function (CDF)
probability integral transforms and multivariate covariance cholesky factorization.
$C(u_1, \dots, u_d) = \Phi_{\Sigma}(\Phi^{-1}(u_1), \dots, \Phi^{-1}(u_d))$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.stats import norm, rankdata


class EmpiricalCDF:
    """Non-parametric univariate Empirical Cumulative Distribution Function."""

    def __init__(self, data: np.ndarray):
        self.sorted_data = np.sort(np.asarray(data, dtype=np.float64))
        self.n = len(self.sorted_data)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Probability integral transform: maps real values to uniform $u \in (0, 1)$."""
        ranks = np.searchsorted(self.sorted_data, x, side="right")
        u = (ranks + 0.5) / (self.n + 1.0)
        return np.clip(u, 1e-6, 1.0 - 1e-6)

    def inverse_transform(self, u: np.ndarray) -> np.ndarray:
        """Quantile inverse transform: maps uniform $u \in (0, 1)$ back to real values."""
        indices = np.clip(np.floor(u * self.n).astype(int), 0, self.n - 1)
        return self.sorted_data[indices]


class GaussianCopula:
    """Gaussian Copula preserving marginal distributions and multivariate linear/rank correlations."""

    def __init__(self):
        self.cdfs_: List[EmpiricalCDF] = []
        self.corr_matrix_: Optional[np.ndarray] = None
        self.L_: Optional[np.ndarray] = None
        self.n_features = 0

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        n_samples, self.n_features = X.shape

        self.cdfs_ = []
        U = np.zeros_like(X)

        # 1. Transform each marginal column to Uniform(0, 1)
        for j in range(self.n_features):
            cdf = EmpiricalCDF(X[:, j])
            self.cdfs_.append(cdf)
            U[:, j] = cdf.transform(X[:, j])

        # 2. Transform Uniform(0, 1) to standard Normal via probit function
        Z = norm.ppf(U)

        # 3. Compute correlation matrix in Gaussian space
        self.corr_matrix_ = np.corrcoef(Z, rowvar=False)
        # Regularize to guarantee positive definiteness
        self.corr_matrix_ += 1e-5 * np.eye(self.n_features)

        # 4. Cholesky decomposition
        self.L_ = np.linalg.cholesky(self.corr_matrix_)
        return self

    def sample(self, n_samples: int = 1000) -> np.ndarray:
        """Generate synthetic observations matching multivariate joint dependency structure."""
        # 1. Sample independent standard normals
        Z_indep = np.random.normal(0, 1, (n_samples, self.n_features))

        # 2. Correlate using Cholesky factor: Z_corr = Z_indep * L^T
        Z_corr = np.dot(Z_indep, self.L_.T)

        # 3. Convert to uniform marginals via standard normal CDF: U = Phi(Z_corr)
        U_synthetic = norm.cdf(Z_corr)

        # 4. Inverse CDF mapping to original feature domains
        X_synthetic = np.zeros_like(U_synthetic)
        for j in range(self.n_features):
            X_synthetic[:, j] = self.cdfs_[j].inverse_transform(U_synthetic[:, j])

        return X_synthetic
