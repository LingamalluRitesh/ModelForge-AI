"""
ModelForge AI - Security: Moments Accountant for Differential Privacy
Implements Abadi, Chu, Goodfellow et al. Deep Learning with Differential Privacy Moments Accountant
providing tight composition bounds over Gaussian Perturbation Mechanisms.
$lpha(\lambda) = \max_{q} \mathbb{E}_{z \sim \mu_0} \left[ \left( rac{\mu_0(z)}{\mu_1(z)} ight)^\lambda ight]$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math
import numpy as np


class MomentsAccountant:
    """Tracks tight privacy loss composition $\epsilon(\delta)$ across $T$ gradient optimization steps."""
    def __init__(self, noise_multiplier: float = 1.1, sampling_ratio: float = 0.01):
        self.sigma = noise_multiplier
        self.q = sampling_ratio
        self.steps = 0
        self.orders = np.arange(1, 33)  # Moments lambda from 1 to 32

    def step(self, steps: int = 1):
        self.steps += steps

    def get_privacy_spent(self, target_delta: float = 1e-5) -> float:
        """
        Calculates bound: $\epsilon = \min_\lambda rac{lpha(\lambda) + \ln(1 / \delta)}{\lambda}$.
        """
        epsilons = []
        for l in self.orders:
            # Upper bound on alpha(lambda) per step: q^2 * lambda * (lambda + 1) / (2 * sigma^2)
            alpha_step = (self.q ** 2) * l * (l + 1) / (2.0 * (self.sigma ** 2))
            alpha_total = self.steps * alpha_step
            eps = (alpha_total + math.log(1.0 / target_delta)) / float(l)
            epsilons.append(eps)

        return float(min(epsilons))
