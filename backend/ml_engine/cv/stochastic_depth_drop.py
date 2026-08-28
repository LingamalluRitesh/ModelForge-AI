"""
ModelForge AI - CV Engine: Stochastic Depth Residual Regularization
Implements Huang et al. Deep Networks with Stochastic Depth
randomly dropping entire residual blocks during training while keeping identity bypass active to prevent gradient vanishing.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class StochasticDepthBlock:
    """Residual block with linear survival probability decay $p_l = 1 - rac{l}{L}(1 - p_L)$."""
    def __init__(self, block_fn: Callable[[np.ndarray], np.ndarray], survival_prob: float = 0.8):
        self.block_fn = block_fn
        self.p_survival = survival_prob

    def forward(self, x: np.ndarray, is_training: bool = True) -> np.ndarray:
        if not is_training:
            # During evaluation scale by survival probability
            return x + self.p_survival * self.block_fn(x)

        # Bernoulli coin flip
        if np.random.rand() < self.p_survival:
            # Keep block active and scale by 1 / p_survival
            return x + (1.0 / self.p_survival) * self.block_fn(x)
        else:
            # Drop block (Identity bypass only)
            return x
