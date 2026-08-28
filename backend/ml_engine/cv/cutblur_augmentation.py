"""
ModelForge AI - CV Engine: CutBlur Image Augmentation
Implements Yoo et al. Rethinking Data Augmentation for Image Super-resolution: A Comprehensive Analysis and a New Strategy
cutting low-resolution / blurred patches and pasting them onto high-resolution images to learn localized de-blurring features.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class CutBlurAugmenter:
    """Replaces random spatial region with downsampled/blurred version."""
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def augment(self, high_res_image: np.ndarray) -> np.ndarray:
        """
        high_res_image: (C, H, W)
        """
        if np.random.rand() > self.prob:
            return high_res_image

        C, H, W = high_res_image.shape
        # Sample box
        bh = np.random.randint(H // 4, H // 2)
        bw = np.random.randint(W // 4, W // 2)
        y = np.random.randint(0, H - bh)
        x = np.random.randint(0, W - bw)

        out = high_res_image.copy()
        patch = high_res_image[:, y : y + bh, x : x + bw]

        # Blur patch via 3x3 uniform box filter
        blurred_patch = np.zeros_like(patch)
        for c in range(C):
            blurred_patch[c] = np.mean(patch[c])

        out[:, y : y + bh, x : x + bw] = blurred_patch
        return out
