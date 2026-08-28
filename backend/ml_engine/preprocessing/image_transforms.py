"""
ModelForge AI - ML Engine: Computer Vision Transforms & Data Augmentation
Implements random affine transforms, random cropping, color jittering, Gaussian blur,
MixUp, CutMix, and automated batch augmentation matrices for image classification.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class ImageAugmentationEngine:
    """Numpy/Scipy based vision data augmentations avoiding deep dependency overhead."""

    @staticmethod
    def random_crop(image: np.ndarray, crop_size: Tuple[int, int]) -> np.ndarray:
        """Random spatial crop of $H \times W \times C$ image."""
        h, w = image.shape[:2]
        ch, cw = crop_size

        if h < ch or w < cw:
            return image

        top = np.random.randint(0, h - ch + 1)
        left = np.random.randint(0, w - cw + 1)
        return image[top : top + ch, left : left + cw]

    @staticmethod
    def random_horizontal_flip(image: np.ndarray, p: float = 0.5) -> np.ndarray:
        """Mirror horizontal flip with probability $p$."""
        if np.random.uniform(0, 1) < p:
            return np.fliplr(image)
        return image

    @staticmethod
    def color_jitter(
        image: np.ndarray,
        brightness: float = 0.2,
        contrast: float = 0.2,
    ) -> np.ndarray:
        """Perturb pixel brightness and contrast."""
        img = image.astype(np.float32)
        # Brightness
        b_factor = 1.0 + np.random.uniform(-brightness, brightness)
        img = img * b_factor

        # Contrast
        c_factor = 1.0 + np.random.uniform(-contrast, contrast)
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        img = (img - mean) * c_factor + mean

        return np.clip(img, 0, 255).astype(image.dtype)

    @staticmethod
    def mixup(
        images1: np.ndarray,
        labels1: np.ndarray,
        images2: np.ndarray,
        labels2: np.ndarray,
        alpha: float = 0.2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Zhang et al. MixUp data augmentation:
        $\tilde{x} = \lambda x_i + (1 - \lambda) x_j$, $\tilde{y} = \lambda y_i + (1 - \lambda) y_j$
        """
        lam = np.random.beta(alpha, alpha)
        mixed_images = lam * images1 + (1.0 - lam) * images2
        mixed_labels = lam * labels1 + (1.0 - lam) * labels2
        return mixed_images, mixed_labels

    @staticmethod
    def cutmix(
        image1: np.ndarray,
        label1: np.ndarray,
        image2: np.ndarray,
        label2: np.ndarray,
        beta: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Yun et al. CutMix region replacement augmentation."""
        h, w = image1.shape[:2]
        lam = np.random.beta(beta, beta)

        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        mixed_image = image1.copy()
        mixed_image[bby1:bby2, bbx1:bbx2] = image2[bby1:bby2, bbx1:bbx2]

        # Adjust lambda to exact area ratio
        adjusted_lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (h * w))
        mixed_label = adjusted_lam * label1 + (1.0 - adjusted_lam) * label2

        return mixed_image, mixed_label
