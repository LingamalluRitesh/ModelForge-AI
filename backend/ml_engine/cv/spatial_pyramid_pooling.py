"""
ModelForge AI - ML Engine: Spatial Pyramid Pooling (SPP-Net)
Implements He et al. Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition
enabling arbitrary image input sizes and generating fixed-length representation vectors.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class SpatialPyramidPooling:
    """Multi-level spatial pooling bins: [4x4, 2x2, 1x1]."""
    def __init__(self, pool_levels: List[int] = [4, 2, 1]):
        self.pool_levels = pool_levels

    def forward(self, conv_features: np.ndarray) -> np.ndarray:
        """
        conv_features: (B, C, H, W)
        returns: (B, C * sum(level^2)) fixed-length representation
        """
        N, C, H, W = conv_features.shape
        pooled_outputs = []

        for level in self.pool_levels:
            # Dynamic kernel size and stride per level
            h_stride = int(np.ceil(H / float(level)))
            w_stride = int(np.ceil(W / float(level)))
            h_kernel = int(np.ceil(H / float(level)))
            w_kernel = int(np.ceil(W / float(level)))

            for i in range(level):
                h_start = i * h_stride
                h_end = min(H, h_start + h_kernel)
                for j in range(level):
                    w_start = j * w_stride
                    w_end = min(W, w_start + w_kernel)

                    bin_data = conv_features[:, :, h_start:h_end, w_start:w_end]
                    bin_max = np.max(bin_data, axis=(2, 3))  # (B, C)
                    pooled_outputs.append(bin_max)

        return np.concatenate(pooled_outputs, axis=-1)
