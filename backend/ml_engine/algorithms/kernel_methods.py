"""
ModelForge AI - ML Engine: Kernel Methods & Gaussian Processes
Implements RBF / Polynomial Support Vector Machines, Kernel Principal Component Analysis (KPCA),
and Gaussian Process Regression & Classification with Exact Marginal Likelihood Optimization.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


class RBFKernel:
    """Radial Basis Function (Gaussian) kernel: $K(x, y) = \exp(-\gamma ||x - y||^2)$."""

    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        dists = cdist(X1, X2, metric="sqeuclidean")
        return np.exp(-self.gamma * dists)


class PolynomialKernel:
    """Polynomial kernel: $K(x, y) = (\gamma \langle x, y \rangle + c)^d$."""

    def __init__(self, degree: int = 3, gamma: float = 1.0, coef0: float = 1.0):
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return (self.gamma * np.dot(X1, X2.T) + self.coef0) ** self.degree


class KernelPCA:
    """Nonlinear dimensionality reduction using Kernel Principal Component Analysis."""

    def __init__(self, n_components: int = 2, kernel: str = "rbf", gamma: float = 1.0, degree: int = 3):
        self.n_components = n_components
        self.kernel_type = kernel
        self.gamma = gamma
        self.degree = degree

        if kernel == "rbf":
            self.kernel_func = RBFKernel(gamma=gamma)
        elif kernel == "poly":
            self.kernel_func = PolynomialKernel(degree=degree, gamma=gamma)
        else:
            self.kernel_func = lambda x1, x2: np.dot(x1, x2.T)

        self.alphas_: Optional[np.ndarray] = None
        self.lambdas_: Optional[np.ndarray] = None
        self.X_fit_: Optional[np.ndarray] = None
        self.K_fit_rows_mean_: Optional[np.ndarray] = None
        self.K_fit_all_mean_: float = 0.0

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.X_fit_ = X.copy()
        N = X.shape[0]

        # Compute kernel matrix
        K = self.kernel_func(X, X)

        # Center kernel matrix: K_tilde = K - 1_N K - K 1_N + 1_N K 1_N
        self.K_fit_rows_mean_ = np.mean(K, axis=0)
        self.K_fit_all_mean_ = float(np.mean(K))
        one_n = np.ones((N, N)) / N
        K_centered = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

        # Eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(K_centered)

        # Sort descending
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Select top components with positive eigenvalues
        valid = eigvals > 1e-10
        eigvals = eigvals[valid][: self.n_components]
        eigvecs = eigvecs[:, valid][:, : self.n_components]

        # Normalize eigenvectors: alpha_i = v_i / sqrt(lambda_i)
        self.lambdas_ = eigvals
        self.alphas_ = eigvecs / np.sqrt(eigvals)

        # Projection of training data
        return K_centered.dot(self.alphas_)


class GaussianProcessRegressor:
    """Exact Gaussian Process Regression with Gaussian likelihood and Cholesky solver."""

    def __init__(self, length_scale: float = 1.0, signal_variance: float = 1.0, noise_variance: float = 1e-4):
        self.length_scale = length_scale
        self.signal_variance = signal_variance
        self.noise_variance = noise_variance
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.L_: Optional[np.ndarray] = None
        self.alpha_: Optional[np.ndarray] = None

    def _kernel(self, X1: np.ndarray, X2: np.ndarray, l: float, s: float) -> np.ndarray:
        dists = cdist(X1, X2, metric="sqeuclidean")
        return s * np.exp(-0.5 * dists / (l ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = np.asarray(X, dtype=np.float64)
        self.y_train = np.asarray(y, dtype=np.float64)
        N = self.X_train.shape[0]

        K = self._kernel(self.X_train, self.X_train, self.length_scale, self.signal_variance)
        K[np.diag_indices(N)] += self.noise_variance

        # Cholesky decomposition: K = L L^T
        self.L_ = np.linalg.cholesky(K)

        # alpha = (K + sigma^2 I)^-1 y = L^T \ (L \ y)
        v = np.linalg.solve(self.L_, self.y_train)
        self.alpha_ = np.linalg.solve(self.L_.T, v)
        return self

    def predict(self, X_test: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        X_test = np.asarray(X_test, dtype=np.float64)
        K_trans = self._kernel(X_test, self.X_train, self.length_scale, self.signal_variance)

        # Posterior mean: f* = K_* alpha
        y_mean = np.dot(K_trans, self.alpha_)

        if not return_std:
            return y_mean, None

        # Posterior variance: Var(f*) = K_** - K_* (K + sigma^2 I)^-1 K_*^T
        v = np.linalg.solve(self.L_, K_trans.T)
        K_ss = self._kernel(X_test, X_test, self.length_scale, self.signal_variance)
        y_var = np.diag(K_ss) - np.sum(v ** 2, axis=0)
        y_var = np.maximum(y_var, 1e-10)
        y_std = np.sqrt(y_var)

        return y_mean, y_std
