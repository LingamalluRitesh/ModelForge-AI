"""
ModelForge AI - ML Engine: Tree-Structured Parzen Estimator (TPE)
Implements Bergstra et al. Tree-Structured Parzen Estimator for Bayesian hyperparameter optimization
using Parzen-Rosenblatt Gaussian kernel density estimation over split response distributions:
$p(x|y) = \ell(x) \text{ if } y < y^* \text{ else } g(x)$, maximizing Expected Improvement $EI(x) = \frac{\ell(x)}{g(x)}$.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.stats import norm


class GaussianKDE:
    """1D Gaussian Kernel Density Estimator with Silverman's Rule-of-Thumb bandwidth."""

    def __init__(self, samples: np.ndarray, bandwidth: Optional[float] = None):
        self.samples = np.asarray(samples, dtype=np.float64)
        n = len(self.samples)
        std = np.std(self.samples) if n > 1 else 1.0
        # Silverman's bandwidth
        self.bandwidth = bandwidth or (1.06 * max(1e-4, std) * (n ** (-1.0 / 5.0)))

    def pdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        diffs = (x[:, np.newaxis] - self.samples[np.newaxis, :]) / self.bandwidth
        kernel_vals = norm.pdf(diffs) / self.bandwidth
        return np.mean(kernel_vals, axis=1)

    def sample(self, n_samples: int = 1) -> np.ndarray:
        chosen_centers = np.random.choice(self.samples, size=n_samples, replace=True)
        noise = np.random.normal(0, self.bandwidth, size=n_samples)
        return chosen_centers + noise


class TreeStructuredParzenEstimator:
    """Non-parametric Bayesian hyperparameter optimization algorithm."""

    def __init__(
        self,
        bounds: List[Tuple[float, float]],
        objective_func: Callable[[np.ndarray], float],
        gamma: float = 0.25,  # Top quantile threshold for l(x)
        n_startup_trials: int = 10,
        n_candidates: int = 64,
        random_state: int = 42,
    ):
        self.bounds = np.array(bounds)
        self.objective_func = objective_func
        self.gamma = gamma
        self.n_startup_trials = n_startup_trials
        self.n_candidates = n_candidates
        self.random_state = random_state
        self.n_dims = len(bounds)

        self.history_x: List[np.ndarray] = []
        self.history_y: List[float] = []

    def optimize(self, n_trials: int = 50) -> Dict[str, Any]:
        np.random.seed(self.random_state)

        for trial in range(n_trials):
            if trial < self.n_startup_trials:
                # Uniform random exploration
                x_next = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
            else:
                x_next = self._sample_next_candidate()

            # Evaluate objective
            y_next = float(self.objective_func(x_next))
            self.history_x.append(x_next)
            self.history_y.append(y_next)

        best_idx = int(np.argmin(self.history_y))
        return {
            "best_params": self.history_x[best_idx].tolist(),
            "best_loss": round(self.history_y[best_idx], 5),
            "total_trials": len(self.history_y),
            "history": [{"params": x.tolist(), "loss": y} for x, y in zip(self.history_x, self.history_y)],
        }

    def _sample_next_candidate(self) -> np.ndarray:
        """Sample candidates from l(x) and pick argmax l(x)/g(x)."""
        X = np.array(self.history_x)
        y = np.array(self.history_y)
        n_samples = len(y)

        # Determine cutoff threshold y*
        cutoff_idx = int(np.ceil(self.gamma * n_samples))
        sorted_indices = np.argsort(y)
        good_indices = sorted_indices[:cutoff_idx]
        bad_indices = sorted_indices[cutoff_idx:]

        X_good = X[good_indices]
        X_bad = X[bad_indices]

        # Generate candidates per dimension from good KDE
        candidates = np.zeros((self.n_candidates, self.n_dims))
        l_densities = np.ones(self.n_candidates)
        g_densities = np.ones(self.n_candidates)

        for d in range(self.n_dims):
            kde_good = GaussianKDE(X_good[:, d])
            kde_bad = GaussianKDE(X_bad[:, d])

            # Sample candidate points from l(x)
            samples_d = kde_good.sample(self.n_candidates)
            samples_d = np.clip(samples_d, self.bounds[d, 0], self.bounds[d, 1])
            candidates[:, d] = samples_d

            l_densities *= kde_good.pdf(samples_d)
            g_densities *= np.maximum(kde_bad.pdf(samples_d), 1e-12)

        # Acquisition ratio: EI(x) proportional to l(x) / g(x)
        ratios = l_densities / g_densities
        best_candidate_idx = np.argmax(ratios)

        return candidates[best_candidate_idx]
