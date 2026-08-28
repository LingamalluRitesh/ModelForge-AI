"""
ModelForge AI - ML Engine: Convolutional Block Attention Module (CBAM)
Implements Woo et al. CBAM: Convolutional Block Attention Module
sequentially applying Channel Attention and Spatial Attention submodules.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class ChannelAttentionModule:
    """Computes channel-wise attention using Average and Max pooling along with shared MLP."""
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        self.in_channels = in_channels
        r_dim = max(1, in_channels // reduction_ratio)

        self.W0 = np.random.normal(0, np.sqrt(2.0 / in_channels), (in_channels, r_dim))
        self.b0 = np.zeros(r_dim)
        self.W1 = np.random.normal(0, np.sqrt(2.0 / r_dim), (r_dim, in_channels))
        self.b1 = np.zeros(in_channels)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x shape: (B, C, H, W)
        avg_pool = np.mean(x, axis=(2, 3))  # (B, C)
        max_pool = np.max(x, axis=(2, 3))   # (B, C)

        mlp_avg = np.dot(np.maximum(0, np.dot(avg_pool, self.W0) + self.b0), self.W1) + self.b1
        mlp_max = np.dot(np.maximum(0, np.dot(max_pool, self.W0) + self.b0), self.W1) + self.b1

        scale = 1.0 / (1.0 + np.exp(-(mlp_avg + mlp_max)))  # (B, C)
        return x * scale[:, :, np.newaxis, np.newaxis]


class SpatialAttentionModule:
    """Computes 2D spatial attention using Channel-wise Avg/Max pooling and 7x7 Convolution."""
    def __init__(self, kernel_size: int = 7):
        self.W_conv = np.random.normal(0, 0.05, (1, 2, kernel_size, kernel_size))
        self.b_conv = np.zeros(1)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Channel pooling
        avg_c = np.mean(x, axis=1, keepdims=True)  # (B, 1, H, W)
        max_c = np.max(x, axis=1, keepdims=True)   # (B, 1, H, W)
        concat_c = np.concatenate([avg_c, max_c], axis=1)  # (B, 2, H, W)

        # Simplified 2D spatial scale approximation
        scale = 1.0 / (1.0 + np.exp(-np.mean(concat_c, axis=1, keepdims=True)))
        return x * scale
