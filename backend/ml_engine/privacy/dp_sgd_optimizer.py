"""
ModelForge AI - ML Engine: Differentially Private Stochastic Gradient Descent (DP-SGD)
Implements Abadi et al. Deep Learning with Differential Privacy via per-sample gradient clipping,
Calibrated Gaussian Noise addition, and Moments Accountant Rényi Differential Privacy (RDP).
$\bar{g} = \frac{1}{B} \left( \sum_{i=1}^B g_i \min\left(1, \frac{C}{||g_i||_2}\right) + \mathcal{N}(0, \sigma^2 C^2 \mathbf{I}) \right)$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class DPSGD:
    """Differentially Private SGD Optimizer."""

    def __init__(
        self,
        lr: float = 0.01,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 1.1,
        target_epsilon: float = 1.0,
        target_delta: float = 1e-5,
    ):
        self.lr = lr
        self.C = max_grad_norm
        self.sigma = noise_multiplier
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta

    def clip_and_perturb(self, per_sample_grads: np.ndarray) -> np.ndarray:
        """
        Clip individual gradients to max L2 norm $C$ and add calibrated Gaussian noise:
        $\tilde{g} = \frac{1}{B} \sum_{i} \text{clip}(g_i, C) + \mathcal{N}(0, \frac{\sigma^2 C^2}{B^2} \mathbf{I})$
        """
        B = per_sample_grads.shape[0]
        grad_norms = np.linalg.norm(per_sample_grads.reshape(B, -1), axis=1)

        # Clipping factors: min(1, C / ||g||)
        clip_factors = np.minimum(1.0, self.C / np.maximum(1e-8, grad_norms))
        clipped_grads = per_sample_grads * clip_factors[:, np.newaxis]

        # Summed clipped gradients
        summed_grad = np.sum(clipped_grads, axis=0)

        # Calibrated Gaussian noise addition
        noise = np.random.normal(0, self.sigma * self.C, size=summed_grad.shape)

        dp_grad = (summed_grad + noise) / B
        return dp_grad
