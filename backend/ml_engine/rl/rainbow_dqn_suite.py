"""
ModelForge AI - ML Engine: Rainbow Deep Q-Network Integration
Implements Hessel et al. Rainbow: Combining Improvements in Deep Reinforcement Learning
integrating Double Q-Learning, Prioritized Replay, Dueling Networks, Multi-Step Returns,
Distributional C51 Categorical RL, and Noisy Exploration Nets.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class NoisyLinear:
    """Fortunato et al. Noisy Networks for Exploration replacing epsilon-greedy with parametric Gaussian noise."""
    def __init__(self, in_features: int, out_features: int, sigma_zero: float = 0.5):
        self.in_features = in_features
        self.out_features = out_features

        # Learnable parameters
        self.mu_w = np.random.uniform(-1.0 / np.sqrt(in_features), 1.0 / np.sqrt(in_features), (in_features, out_features))
        self.sigma_w = np.full((in_features, out_features), sigma_zero / np.sqrt(in_features))

        self.mu_b = np.random.uniform(-1.0 / np.sqrt(in_features), 1.0 / np.sqrt(in_features), out_features)
        self.sigma_b = np.full(out_features, sigma_zero / np.sqrt(in_features))

    def _f(self, x: np.ndarray) -> np.ndarray:
        return np.sign(x) * np.sqrt(np.abs(x))

    def forward(self, x: np.ndarray, evaluate: bool = False) -> np.ndarray:
        if evaluate:
            return np.dot(x, self.mu_w) + self.mu_b

        # Factorized Gaussian noise
        p = np.random.normal(0, 1, self.in_features)
        q = np.random.normal(0, 1, self.out_features)
        eps_w = np.outer(self._f(p), self._f(q))
        eps_b = self._f(q)

        W = self.mu_w + self.sigma_w * eps_w
        b = self.mu_b + self.sigma_b * eps_b
        return np.dot(x, W) + b


class CategoricalC51Head:
    """Bellemare et al. Distributional RL projecting Value Distribution across $N=51$ discrete probability atoms."""
    def __init__(self, in_features: int, n_actions: int, num_atoms: int = 51, v_min: float = -10.0, v_max: float = 10.0):
        self.n_actions = n_actions
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.support = np.linspace(v_min, v_max, num_atoms)
        self.delta_z = (v_max - v_min) / (num_atoms - 1)

        self.head = NoisyLinear(in_features, n_actions * num_atoms)

    def forward(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns categorical probabilities $(B, n\_actions, num\_atoms)$ and expected Q-values $(B, n\_actions)$.
        """
        N = features.shape[0]
        logits = self.head.forward(features).reshape(N, self.n_actions, self.num_atoms)

        # Softmax over atom dimension
        shift = logits - np.max(logits, axis=-1, keepdims=True)
        probs = np.exp(shift) / np.sum(np.exp(shift), axis=-1, keepdims=True)

        # Expected Q-value: sum_i p_i * z_i
        expected_q = np.sum(probs * self.support, axis=-1)
        return probs, expected_q
