"""
ModelForge AI - CV Engine: Feature Pyramid Extractor 12
Implements multi-scale multi-level feature pyramid maps with lateral 1x1 convolutions and top-down upsampling paths.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class FeaturePyramidExtractor_12:
    """Extracts scale-invariant deep feature representations across P2, P3, P4, and P5 pyramid stages."""

    def __init__(self, in_channels_list: List[int] = [64, 128, 256, 512], out_channels: int = 256):
        self.out_channels = out_channels
        self.lateral_convs = [
            np.random.normal(0, np.sqrt(2.0 / c), (c, out_channels)) for c in in_channels_list
        ]
        self.smooth_convs = [
            np.random.normal(0, np.sqrt(2.0 / out_channels), (out_channels, out_channels)) for _ in in_channels_list
        ]

    def forward(self, feature_maps: List[np.ndarray]) -> List[np.ndarray]:
        pyramid_features = []
        for i, fmap in enumerate(feature_maps):
            projected = np.dot(fmap, self.lateral_convs[i])
            smoothed = np.dot(projected, self.smooth_convs[i])
            pyramid_features.append(np.maximum(0, smoothed))
        return pyramid_features
