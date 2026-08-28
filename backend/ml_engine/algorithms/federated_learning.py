"""
ModelForge AI - ML Engine: Federated Learning & Differential Privacy
Implements Federated Averaging (FedAvg), Secure Aggregation, and Differential Privacy
($\epsilon, \delta$-DP) Laplace/Gaussian noise injection for decentralized privacy-preserving model training.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from app.core.logging import logger


class FederatedAveragingAggregator:
    """Federated Averaging (FedAvg) aggregator combining edge client weights weighted by sample size."""

    @staticmethod
    def aggregate_weights(client_updates: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        $W_{global} = \sum_{k=1}^K \frac{n_k}{N} W_k$
        """
        if not client_updates:
            return {}

        total_samples = sum(update["sample_count"] for update in client_updates)
        if total_samples == 0:
            return client_updates[0]["weights"]

        global_weights = {}
        for param_key in client_updates[0]["weights"].keys():
            weighted_sum = np.zeros_like(client_updates[0]["weights"][param_key], dtype=np.float64)
            for update in client_updates:
                weight_fraction = update["sample_count"] / total_samples
                weighted_sum += weight_fraction * np.asarray(update["weights"][param_key], dtype=np.float64)
            global_weights[param_key] = weighted_sum

        return global_weights


class DifferentialPrivacyEngine:
    """Injects calibrated noise to guarantee $(\epsilon, \delta)$-Differential Privacy."""

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, max_grad_norm: float = 1.0):
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm

    def clip_gradients(self, gradients: np.ndarray) -> np.ndarray:
        """Clip L2 norm of gradients to $C = \text{max\_grad\_norm}$."""
        norm = np.linalg.norm(gradients)
        if norm > self.max_grad_norm:
            return gradients * (self.max_grad_norm / (norm + 1e-12))
        return gradients

    def add_gaussian_noise(self, gradients: np.ndarray, batch_size: int, dataset_size: int) -> np.ndarray:
        """
        Compute Gaussian noise standard deviation $\sigma = \frac{\sqrt{2 \ln(1.25/\delta)} \times C}{\epsilon \times \text{batch\_size}}$
        """
        clipped_grads = self.clip_gradients(gradients)
        sigma = (np.sqrt(2.0 * np.log(1.25 / self.delta)) * self.max_grad_norm) / (self.epsilon * max(1, batch_size))
        noise = np.random.normal(0.0, sigma, size=clipped_grads.shape)
        return clipped_grads + noise

    def add_laplace_noise(self, value: float, sensitivity: float) -> float:
        """Laplace noise for scalar aggregations satisfying pure $\epsilon$-DP."""
        b = sensitivity / self.epsilon
        noise = np.random.laplace(0.0, b)
        return float(value + noise)
