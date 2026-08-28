"""
ModelForge AI - ML Engine: xDeepFM Compressed Interaction Network (CIN)
Implements Lian et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems
with vector-wise polynomial feature interaction mapping in tensor outer product space.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class CompressedInteractionLayer:
    """Computes vector-wise explicit feature interaction tensor $X^k$."""
    def __init__(self, in_fields_h: int, num_fields_m: int, num_filters: int):
        self.in_h = in_fields_h
        self.m = num_fields_m
        self.num_filters = num_filters

        std = np.sqrt(2.0 / (in_fields_h * num_fields_m))
        self.W = np.random.normal(0, std, (num_filters, in_fields_h, num_fields_m))
        self.b = np.zeros(num_filters)

    def forward(self, X0: np.ndarray, Xk: np.ndarray) -> np.ndarray:
        """
        X0: (B, m, D) initial embedding matrix
        Xk: (B, H_k, D) feature map from k-th layer
        returns: (B, num_filters, D)
        """
        B, H_k, D = Xk.shape
        # Outer product: Z shape (B, D, H_k, m)
        Z = np.einsum("bhd,bmd->bdhm", Xk, X0)

        # 1D Convolution over (H_k, m) planes
        out = np.einsum("bdhm,fhm->bfd", Z, self.W)
        out = out + self.b[np.newaxis, :, np.newaxis]
        return np.maximum(0, out)
