"""
ModelForge AI - CV Engine: Focal Loss for Dense Object Detection
Implements Lin et al. Focal Loss for Dense Object Detection
dynamically down-weighting the loss assigned to easy background examples.
$	ext{FL}(p_t) = -lpha_t (1 - p_t)^\gamma \log(p_t)$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class FocalLoss:
    """Focal Loss with focusing parameter gamma and balancing alpha."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        # Sigmoid probabilities
        p = 1.0 / (1.0 + np.exp(-logits))
        p = np.clip(p, 1e-7, 1.0 - 1e-7)

        # p_t: probability of true class
        p_t = np.where(targets == 1, p, 1.0 - p)
        alpha_t = np.where(targets == 1, self.alpha, 1.0 - self.alpha)

        # Focal modulating factor: (1 - p_t)^gamma
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss = -alpha_t * modulating_factor * np.log(p_t)
        return float(np.mean(loss))
