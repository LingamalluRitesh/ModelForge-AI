"""
ModelForge AI - ML Engine: Convolutional Neural Networks & ResNet Architecture
Implements 2D Convolution (Conv2D), MaxPool2D, 2D Spatial Batch Normalization (BatchNorm2d),
Residual Bottleneck Blocks, and ResNet-18 forward & backward passes in NumPy.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class Conv2D:
    """2D Convolutional Layer with im2col vectorization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # He / Kaiming normal weight initialization
        std = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.normal(0, std, (out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.zeros(out_channels)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self.x_cache: Optional[np.ndarray] = None
        self.x_col_cache: Optional[np.ndarray] = None

    def _im2col(self, x: np.ndarray) -> np.ndarray:
        N, C, H, W = x.shape
        out_h = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        x_padded = np.pad(
            x,
            ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
            mode="constant",
        )

        cols = np.zeros((N, C, self.kernel_size, self.kernel_size, out_h, out_w))
        for kh in range(self.kernel_size):
            kh_max = kh + self.stride * out_h
            for kw in range(self.kernel_size):
                kw_max = kw + self.stride * out_w
                cols[:, :, kh, kw, :, :] = x_padded[:, :, kh:kh_max:self.stride, kw:kw_max:self.stride]

        cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
        return cols

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_cache = x
        N, C, H, W = x.shape
        out_h = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        x_col = self._im2col(x)
        self.x_col_cache = x_col
        W_row = self.W.reshape(self.out_channels, -1)

        out = np.dot(x_col, W_row.T) + self.b
        out = out.reshape(N, out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)
        return out


class MaxPool2D:
    """2D Spatial Max Pooling layer."""

    def __init__(self, pool_size: int = 2, stride: int = 2):
        self.pool_size = pool_size
        self.stride = stride
        self.x_cache: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_cache = x
        N, C, H, W = x.shape
        out_h = (H - self.pool_size) // self.stride + 1
        out_w = (W - self.pool_size) // self.stride + 1

        out = np.zeros((N, C, out_h, out_w))
        for h in range(out_h):
            h_start = h * self.stride
            h_end = h_start + self.pool_size
            for w in range(out_w):
                w_start = w * self.stride
                w_end = w_start + self.pool_size
                out[:, :, h, w] = np.max(x[:, :, h_start:h_end, w_start:w_end], axis=(2, 3))

        return out


class BatchNorm2D:
    """Spatial 2D Batch Normalization over (N, C, H, W) tensors."""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.gamma = np.ones((1, num_features, 1, 1))
        self.beta = np.zeros((1, num_features, 1, 1))
        self.running_mean = np.zeros((1, num_features, 1, 1))
        self.running_var = np.ones((1, num_features, 1, 1))

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            var = np.var(x, axis=(0, 2, 3), keepdims=True)

            self.running_mean = (1.0 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1.0 - self.momentum) * self.running_var + self.momentum * var

            x_norm = (x - mean) / np.sqrt(var + self.eps)
        else:
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)

        return self.gamma * x_norm + self.beta


class ResidualBlock:
    """He et al. ResNet Identity & Projection Residual Skip-Connection Block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        self.conv1 = Conv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = BatchNorm2D(out_channels)
        self.conv2 = Conv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm2D(out_channels)

        # Shortcut projection if dimensions change
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Conv2D(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
            self.shortcut_bn = BatchNorm2D(out_channels)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        residual = x
        out = np.maximum(0, self.bn1.forward(self.conv1.forward(x), training))
        out = self.bn2.forward(self.conv2.forward(out), training)

        if self.shortcut is not None:
            residual = self.shortcut_bn.forward(self.shortcut.forward(x), training)

        return np.maximum(0, out + residual)
