"""
ModelForge AI - AutoML: Tree-Structured Parzen Estimator (TPE) Sampler
Implements Bergstra et al. Algorithms for Hyper-Parameter Optimization (TPE)
modeling good and bad hyperparameter partitions with Kernel Density Estimation (KDE).
$	ext{EI}(x) = rac{\gamma y^* \ell(x) - \ell(x) \int_{-\infty}^{y^*} p(y) dy}{\gamma \ell(x) + (1 - \gamma) g(x)} \propto rac{\ell(x)}{g(x)}$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class TreeStructuredParzenEstimator:
    """Acquisition-driven TPE optimizer maximizing Expected Improvement (EI)."""
    def __init__(self, parameter_ranges: Dict[str, Tuple[float, float]], gamma: float = 0.15):
        self.ranges = parameter_ranges
        self.gamma = gamma
        self.history: List[Tuple[Dict[str, float], float]] = []

    def observe(self, params: Dict[str, float], loss: float):
        self.history.append((params, loss))

    def sample_next_candidate(self, n_ei_candidates: int = 24) -> Dict[str, float]:
        if len(self.history) < 10:
            # Random exploration initialization
            candidate = {}
            for name, (low, high) in self.ranges.items():
                candidate[name] = float(np.random.uniform(low, high))
            return candidate

        # Split history into top gamma quantile (good) and remaining (bad)
        sorted_trials = sorted(self.history, key=lambda item: item[1])
        n_good = max(2, int(len(sorted_trials) * self.gamma))
        good_trials = [t[0] for t in sorted_trials[:n_good]]
        bad_trials = [t[0] for t in sorted_trials[n_good:]]

        best_candidate = None
        max_ratio = -float("inf")

        # Generate candidates from good trials with Gaussian perturbation
        for _ in range(n_ei_candidates):
            parent = good_trials[np.random.randint(len(good_trials))]
            cand = {}
            for name, (low, high) in self.ranges.items():
                std = 0.15 * (high - low)
                val = float(np.clip(np.random.normal(parent[name], std), low, high))
                cand[name] = val

            # Ratio calculation l(x) / g(x)
            l_score = sum(np.exp(-0.5 * ((cand[k] - g[k]) ** 2)) for g in good_trials) / float(len(good_trials))
            g_score = sum(np.exp(-0.5 * ((cand[k] - b[k]) ** 2)) for b in bad_trials) / float(len(bad_trials))

            ratio = l_score / max(1e-8, g_score)
            if ratio > max_ratio:
                max_ratio = ratio
                best_candidate = cand

        return best_candidate or cand
