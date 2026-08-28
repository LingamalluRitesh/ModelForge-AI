"""
ModelForge AI - ML Engine: Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
Implements Hansen Completely Derandomized Self-Adaptation in Evolution Strategies (CMA-ES)
with Rank-$\mu$ and Rank-1 covariance matrix adaptation, cumulative step-size adaptation (CSA),
and evolutionary path tracking.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class CMAESOptimizer:
    """State-of-the-art derivative-free continuous black-box hyperparameter optimizer."""
    def __init__(
        self,
        dimension: int,
        objective_fn: Callable[[np.ndarray], float],
        sigma_init: float = 0.5,
        pop_size: Optional[int] = None,
    ):
        self.N = dimension
        self.obj_fn = objective_fn
        self.sigma = sigma_init

        # Population size lambda and parent size mu
        self.lam = pop_size or int(4 + np.floor(3 * np.log(self.N)))
        self.mu = self.lam // 2

        # Recombination weights
        weights_raw = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights_raw / np.sum(weights_raw)
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)

        # Adaptation parameters
        self.cc = (4.0 + self.mu_eff / self.N) / (self.N + 4.0 + 2.0 * self.mu_eff / self.N)
        self.cs = (self.mu_eff + 2.0) / (self.N + self.mu_eff + 5.0)
        self.c1 = 2.0 / ((self.N + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(1.0 - self.c1, 2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff) / ((self.N + 2.0) ** 2 + self.mu_eff))
        self.damps = 1.0 + 2.0 * max(0.0, np.sqrt((self.mu_eff - 1.0) / (self.N + 1.0)) - 1.0) + self.cs
        self.chi_n = np.sqrt(self.N) * (1.0 - 1.0 / (4.0 * self.N) + 1.0 / (21.0 * (self.N ** 2)))

        # Dynamic state
        self.m = np.random.uniform(-1, 1, self.N)
        self.p_c = np.zeros(self.N)
        self.p_s = np.zeros(self.N)
        self.C = np.eye(self.N)

    def optimize(self, n_generations: int = 40) -> Dict[str, Any]:
        best_x = self.m.copy()
        best_loss = float("inf")

        for gen in range(n_generations):
            # Eigen-decomposition of covariance matrix C = B D^2 B^T
            eigvals, B = np.linalg.eigh(self.C)
            eigvals = np.maximum(1e-10, eigvals)
            D = np.sqrt(eigvals)

            # Sample lambda offspring candidates: x_k = m + sigma * B * D * z_k
            z_samples = np.random.normal(0, 1, (self.lam, self.N))
            y_samples = np.array([np.dot(B, D * z) for z in z_samples])
            x_samples = np.array([self.m + self.sigma * y for y in y_samples])

            # Evaluate objective
            scores = np.array([self.obj_fn(x) for x in x_samples])
            order = np.argsort(scores)

            if scores[order[0]] < best_loss:
                best_loss = float(scores[order[0]])
                best_x = x_samples[order[0]].copy()

            # Selection and recombination
            selected_y = y_samples[order[: self.mu]]
            y_w = np.sum(self.weights[:, np.newaxis] * selected_y, axis=0)
            self.m += self.sigma * y_w

            # Step-size adaptation path
            inv_sqrt_c = np.dot(B, (1.0 / D) * np.dot(B.T, y_w))
            self.p_s = (1.0 - self.cs) * self.p_s + np.sqrt(self.cs * (2.0 - self.cs) * self.mu_eff) * inv_sqrt_c
            norm_ps = np.linalg.norm(self.p_s)

            # Covariance path
            h_sig = 1.0 if (norm_ps / np.sqrt(1.0 - (1.0 - self.cs) ** (2 * (gen + 1)))) < (1.4 + 2.0 / (self.N + 1.0)) * self.chi_n else 0.0
            self.p_c = (1.0 - self.cc) * self.p_c + h_sig * np.sqrt(self.cc * (2.0 - self.cc) * self.mu_eff) * y_w

            # Covariance matrix update
            rank_1 = np.outer(self.p_c, self.p_c)
            rank_mu = np.zeros((self.N, self.N))
            for i in range(self.mu):
                rank_mu += self.weights[i] * np.outer(selected_y[i], selected_y[i])

            self.C = (1.0 - self.c1 - self.cmu) * self.C + self.c1 * rank_1 + self.cmu * rank_mu

            # Update step-size sigma
            self.sigma *= np.exp((self.cs / self.damps) * (norm_ps / self.chi_n - 1.0))

        return {
            "best_params": best_x.tolist(),
            "best_loss": round(best_loss, 5),
            "final_sigma": round(self.sigma, 4),
        }
