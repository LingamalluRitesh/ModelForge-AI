"""
ModelForge AI - ML Engine: Disjoint & Hybrid LinUCB Contextual Bandits
Implements Li et al. A Contextual-Bandit Approach to Personalized Recommendation (LinUCB)
with Ridge regression confidence bounds and Sherman-Morrison rank-1 matrix inverse updates.
$a_t = \arg\max_{a \in \mathcal{A}} \left( \hat{\theta}_a^T x_{t,a} + \alpha \sqrt{x_{t,a}^T A_a^{-1} x_{t,a}} \right)$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class DisjointLinUCB:
    """Disjoint Linear Upper Confidence Bound Contextual Bandit."""

    def __init__(self, n_arms: int, feature_dim: int, alpha: float = 1.0):
        self.n_arms = n_arms
        self.d = feature_dim
        self.alpha = alpha

        # Per-arm design matrix A_a (d x d) and response vector b_a (d)
        self.A = [np.eye(self.d) for _ in range(n_arms)]
        self.A_inv = [np.eye(self.d) for _ in range(n_arms)]
        self.b = [np.zeros((self.d, 1)) for _ in range(n_arms)]

    def select_arm(self, context: np.ndarray) -> int:
        """Select arm maximizing UCB score: $\hat{\theta}_a^T x + \alpha \sqrt{x^T A_a^{-1} x}$."""
        x = np.asarray(context, dtype=np.float64).reshape(-1, 1)
        p = np.zeros(self.n_arms)

        for a in range(self.n_arms):
            theta_a = np.dot(self.A_inv[a], self.b[a])
            var = float(np.dot(x.T, np.dot(self.A_inv[a], x)))
            sd = np.sqrt(max(0.0, var))
            expected_payoff = float(np.dot(theta_a.T, x))
            p[a] = expected_payoff + self.alpha * sd

        return int(np.argmax(p))

    def update(self, chosen_arm: int, context: np.ndarray, reward: float):
        """Update arm covariance matrix using Sherman-Morrison rank-1 formula."""
        x = np.asarray(context, dtype=np.float64).reshape(-1, 1)

        # Update A and b
        self.A[chosen_arm] += np.dot(x, x.T)
        self.b[chosen_arm] += reward * x

        # Sherman-Morrison update: $(A + x x^T)^{-1} = A^{-1} - \frac{A^{-1} x x^T A^{-1}}{1 + x^T A^{-1} x}$
        A_inv = self.A_inv[chosen_arm]
        v = np.dot(A_inv, x)
        denom = 1.0 + float(np.dot(x.T, v))
        self.A_inv[chosen_arm] = A_inv - np.dot(v, v.T) / denom
