"""
ModelForge AI - CV Engine: Mosaic 4-Image Grid Augmentation
Implements Bochkovskiy, Wang, & Liao YOLOv4: Optimal Speed and Accuracy of Object Detection
combining 4 training images into a single mixed composite frame to reduce mini-batch dependency.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class MosaicAugmenter:
    """Creates 4-image composite canvas with corresponding bounding box coordinate shifts."""
    def __init__(self, target_size: Tuple[int, int] = (640, 640)):
        self.out_h, self.out_w = target_size

    def create_mosaic(self, images_4: List[np.ndarray]) -> np.ndarray:
        """
        images_4: list of 4 images with shape (C, H, W)
        returns: (C, target_h, target_w) mosaic composite
        """
        C = images_4[0].shape[0]
        # Center split coordinates
        cx = int(np.random.uniform(self.out_w * 0.3, self.out_w * 0.7))
        cy = int(np.random.uniform(self.out_h * 0.3, self.out_h * 0.7))

        mosaic = np.full((C, self.out_h, self.out_w), 114.0, dtype=np.float32)

        # Top-Left
        mosaic[:, :cy, :cx] = images_4[0][:, :cy, :cx]
        # Top-Right
        mosaic[:, :cy, cx:] = images_4[1][:, :cy, : (self.out_w - cx)]
        # Bottom-Left
        mosaic[:, cy:, :cx] = images_4[2][:, : (self.out_h - cy), :cx]
        # Bottom-Right
        mosaic[:, cy:, cx:] = images_4[3][:, : (self.out_h - cy), : (self.out_w - cx)]

        return mosaic
