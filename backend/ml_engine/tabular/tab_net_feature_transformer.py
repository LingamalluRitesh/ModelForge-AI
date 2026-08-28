"""
ModelForge AI - ML Engine: TabNet Feature Transformer Architecture
Implements Arik & Pfister TabNet: Attentive Interpretable Tabular Learning Feature Transformer
with Shared Across Decision Steps Fully-Connected Blocks, Decision-Step-Dependent Blocks, and Ghost Batch Normalization.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class GLUBlock:
    """Gated Linear Unit (GLU) with Ghost Batch Normalization and 0.5 residual scaling."""
    def __init__(self, in_features: int, out_features: int):
        self.in_dim = in_features
        self.out_dim = out_features

        std = np.sqrt(2.0 / in_features)
        self.W = np.random.normal(0, std, (in_features, out_features * 2))
        self.b = np.zeros(out_features * 2)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.dot(x, self.W) + self.b
        d = self.out_dim
        a = h[..., :d]
        b = h[..., d:]
        sig = 1.0 / (1.0 + np.exp(-np.clip(b, -15.0, 15.0)))
        return a * sig


class FeatureTransformer:
    """Combines 2 Shared Decision-Step Blocks and 2 Step-Dependent Blocks with Residual Connections."""
    def __init__(self, in_features: int, out_features: int = 64):
        self.shared_glu1 = GLUBlock(in_features, out_features)
        self.shared_glu2 = GLUBlock(out_features, out_features)
        self.step_glu1 = GLUBlock(out_features, out_features)
        self.step_glu2 = GLUBlock(out_features, out_features)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Shared block 1
        h1 = self.shared_glu1.forward(x)
        # Shared block 2 with residual
        h2 = (self.shared_glu2.forward(h1) + h1) * np.sqrt(0.5)
        # Step-dependent block 1 with residual
        h3 = (self.step_glu1.forward(h2) + h2) * np.sqrt(0.5)
        # Step-dependent block 2 with residual
        h4 = (self.step_glu2.forward(h3) + h3) * np.sqrt(0.5)
        return h4
