"""
ModelForge AI - ML Engine: Gradient-Weighted Class Activation Mapping (Grad-CAM)
Implements Selvaraju et al. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
computing neuron importance weights $lpha_k^c = rac{1}{Z} \sum_i \sum_j rac{\partial y^c}{\partial A_{i,j}^k}$
and coarse localization heatmaps $L_{	ext{Grad-CAM}}^c = 	ext{ReLU}\left(\sum_k lpha_k^c A^kight)$.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class GradCAMEngine:
    """Produces 2D visual feature activation heatmaps for arbitrary convolutional backbones."""
    def __init__(self, target_layer_name: str = "conv5"):
        self.target_layer_name = target_layer_name

    def generate_heatmap(
        self,
        feature_maps: np.ndarray,
        gradients: np.ndarray,
    ) -> np.ndarray:
        """
        feature_maps: (C, H, W) activations of target convolutional layer
        gradients: (C, H, W) gradient of target class score with respect to feature maps
        returns: (H, W) normalized 2D localization heatmap in [0, 1]
        """
        # Global Average Pooling over spatial dimensions (H, W) to obtain neuron weights alpha_k
        alpha_k = np.mean(gradients, axis=(1, 2))  # (C,)

        # Linear combination of forward feature maps weighted by alpha
        cam = np.sum(alpha_k[:, np.newaxis, np.newaxis] * feature_maps, axis=0)

        # Apply ReLU to retain only positive features influencing the target class
        cam = np.maximum(0.0, cam)

        # Normalize to [0, 1]
        max_val = np.max(cam)
        if max_val > 0:
            cam /= max_val

        return cam
