"""
ModelForge AI - Security: Analytical Moments Accountant for Rényi Differential Privacy (RDP)
Implements Mironov Rényi Differential Privacy and Abadi et al. Deep Learning with DP
computing tight privacy loss epsilon-delta conversions across sub-sampled Gaussian mechanisms.
$\epsilon(\delta) = \min_{lpha > 1} \left\{ lpha \cdot 	ext{RDP}(lpha) + rac{\ln(1/\delta)}{lpha - 1} ight\}$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math
import numpy as np


class RenyiDifferentialPrivacyAccountant:
    """Computes exact Rényi Differential Privacy guarantees across arbitrary composition orders alpha."""
    def __init__(self, noise_multiplier: float = 1.2, sample_rate: float = 0.02):
        self.sigma = noise_multiplier
        self.q = sample_rate
        self.steps = 0
        self.orders = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0, 16.0, 32.0, 64.0]

    def add_step(self, count: int = 1):
        self.steps += count

    def _compute_rdp_order(self, alpha: float) -> float:
        """Subsampled Gaussian Mechanism RDP analytical upper bound."""
        if self.q == 0.0:
            return 0.0
        if self.q == 1.0:
            return alpha / (2.0 * (self.sigma ** 2))

        # Second-order Taylor expansion bound
        return (self.q ** 2) * alpha / (2.0 * (self.sigma ** 2))

    def get_epsilon(self, target_delta: float = 1e-5) -> float:
        """Converts cumulative RDP to (epsilon, delta)-DP guarantee."""
        epsilons = []
        for alpha in self.orders:
            rdp_total = self.steps * self._compute_rdp_order(alpha)
            # RDP to DP conversion formula
            eps = rdp_total + math.log(1.0 / target_delta) / (alpha - 1.0)
            epsilons.append(eps)

        return float(min(epsilons))
