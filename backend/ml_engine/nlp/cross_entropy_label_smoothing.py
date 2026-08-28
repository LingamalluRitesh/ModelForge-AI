"""
ModelForge AI - Preprocessing: Label Smoothing Regularization Loss
Implements Müller, Kornblith, & Hinton When Does Label Smoothing Help?
preventing overconfident soft target logit predictions and improving probability calibration.
$y_{LS} = (1 - lpha) y_{one-hot} + rac{lpha}{K}$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class LabelSmoothingLoss:
    """Computes Cross-Entropy with uniform label smoothing noise."""
    def __init__(self, alpha: float = 0.1, num_classes: int = 10):
        self.alpha = alpha
        self.K = num_classes

    def smooth_targets(self, targets: np.ndarray) -> np.ndarray:
        N = len(targets)
        one_hot = np.zeros((N, self.K))
        one_hot[np.arange(N), targets] = 1.0
        return (1.0 - self.alpha) * one_hot + (self.alpha / float(self.K))

    def compute_loss(self, logits: np.ndarray, targets: np.ndarray) -> float:
        shift = logits - np.max(logits, axis=-1, keepdims=True)
        log_probs = shift - np.log(np.sum(np.exp(shift), axis=-1, keepdims=True))
        y_smoothed = self.smooth_targets(targets)
        loss = -np.mean(np.sum(y_smoothed * log_probs, axis=-1))
        return float(loss)
