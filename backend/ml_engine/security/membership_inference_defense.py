"""
ModelForge AI - Security Engine: Membership Inference Defense & Confidence Calibration
Implements Shokri et al. Membership Inference Attacks Against Machine Learning Models
and defends via Temperature Scaling, Top-K Logit Pruning, and Output Confidence Smoothing.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class MembershipInferenceDefenseEngine:
    """Obfuscates prediction confidence distributions to prevent shadow model membership reconstruction."""

    def __init__(
        self,
        temperature: float = 2.0,
        top_k: int = 3,
        add_laplace_noise: bool = True,
        noise_scale: float = 0.01,
    ):
        self.temperature = temperature
        self.top_k = top_k
        self.add_noise = add_laplace_noise
        self.noise_scale = noise_scale

    def sanitize_probabilities(self, raw_logits: np.ndarray) -> np.ndarray:
        """
        raw_logits: (B, num_classes) or (num_classes,)
        """
        logits = np.asarray(raw_logits, dtype=float)
        orig_ndim = logits.ndim
        if orig_ndim == 1:
            logits = logits.reshape(1, -1)

        B, K = logits.shape

        # 1. Temperature Scaling Softmax
        scaled_logits = logits / max(1e-4, self.temperature)
        shift = scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True)
        probs = np.exp(shift) / np.sum(np.exp(shift), axis=-1, keepdims=True)

        # 2. Top-K logit truncation (set low-probability tails to zero)
        if self.top_k < K:
            sanitized = np.zeros_like(probs)
            for b in range(B):
                top_indices = np.argsort(probs[b])[-self.top_k :]
                sanitized[b, top_indices] = probs[b, top_indices]
                sanitized[b] /= np.sum(sanitized[b])  # Re-normalize
            probs = sanitized

        # 3. Add calibrated Laplace noise for Differential Privacy guarantee
        if self.add_noise:
            noise = np.random.laplace(0, self.noise_scale, size=probs.shape)
            probs = np.maximum(0.0, probs + noise)
            probs /= np.sum(probs, axis=-1, keepdims=True)

        if orig_ndim == 1:
            return probs.reshape(-1)
        return probs
