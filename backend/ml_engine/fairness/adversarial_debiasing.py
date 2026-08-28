"""
ModelForge AI - Fairness: Adversarial Debiasing Representation Engine
Implements Zhang, Lemoine, & Mitchell Mitigating Unwanted Biases with Adversarial Learning
simultaneously training a primary predictor and an adversarial adversary attempting to predict protected attributes.
$\min_{W} \max_{U} \mathcal{L}_{pred}(f(X; W), Y) - \lambda \mathcal{L}_{adv}(g(f(X; W); U), Z)$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class AdversarialDebiaser:
    """Trains fair feature representations invariant to sensitive attributes $Z$."""
    def __init__(self, in_features: int, hidden_dim: int = 32, lambda_adv: float = 0.5):
        self.lambda_adv = lambda_adv

        # Predictor Network (W)
        std_p = np.sqrt(2.0 / in_features)
        self.W_p1 = np.random.normal(0, std_p, (in_features, hidden_dim))
        self.b_p1 = np.zeros(hidden_dim)
        self.W_p2 = np.random.normal(0, np.sqrt(2.0 / hidden_dim), (hidden_dim, 1))
        self.b_p2 = np.zeros(1)

        # Adversary Network (U)
        self.W_a1 = np.random.normal(0, np.sqrt(2.0 / hidden_dim), (hidden_dim, 16))
        self.b_a1 = np.zeros(16)
        self.W_a2 = np.random.normal(0, np.sqrt(2.0 / 16), (16, 1))
        self.b_a2 = np.zeros(1)

    def forward_representation(self, X: np.ndarray) -> np.ndarray:
        return np.maximum(0, np.dot(X, self.W_p1) + self.b_p1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        h = self.forward_representation(X)
        logits = np.dot(h, self.W_p2) + self.b_p2
        return 1.0 / (1.0 + np.exp(-logits))
