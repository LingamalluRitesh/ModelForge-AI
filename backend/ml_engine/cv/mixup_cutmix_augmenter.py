"""
ModelForge AI - Preprocessing: Mixup & CutMix Computer Vision Augmentation
Implements Zhang et al. Mixup: Beyond Empirical Risk Minimization and Yun et al. CutMix
regularizing deep convolutional and Vision Transformer backbones against memorization.
$	ilde{x} = \lambda x_i + (1 - \lambda) x_j$, $	ilde{y} = \lambda y_i + (1 - \lambda) y_j$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class MixupCutMixAugmenter:
    """Applies stochastic continuous interpolation and bounding box patch replacement."""
    def __init__(self, alpha_mixup: float = 0.8, alpha_cutmix: float = 1.0, prob_cutmix: float = 0.5):
        self.alpha_mix = alpha_mixup
        self.alpha_cut = alpha_cutmix
        self.prob_cutmix = prob_cutmix

    def apply_mixup(self, x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha_mix, self.alpha_mix)
        x_mixed = lam * x1 + (1.0 - lam) * x2
        y_mixed = lam * y1 + (1.0 - lam) * y2
        return x_mixed, y_mixed

    def apply_cutmix(self, x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        C, H, W = x1.shape
        lam = np.random.beta(self.alpha_cut, self.alpha_cut)

        # Compute bounding box dimensions
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        # Replace patch
        x_cut = x1.copy()
        x_cut[:, bby1:bby2, bbx1:bbx2] = x2[:, bby1:bby2, bbx1:bbx2]

        # Adjust lambda to exact pixel ratio
        adj_lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / float(W * H))
        y_cut = adj_lam * y1 + (1.0 - adj_lam) * y2
        return x_cut, y_cut
