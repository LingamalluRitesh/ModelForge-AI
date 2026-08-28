"""
ModelForge AI - ML Engine: Soft Actor-Critic (SAC) Maximum Entropy Reinforcement Learning
Implements Haarnoja et al. Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning
with Dual Q-Networks, Squashed Gaussian Policy, and Automatic Entropy Temperature alpha adjustment.
$J(\pi) = \sum_{t=0}^T \mathbb{E}_{(s_t, a_t) \sim ho_\pi} \left[ r(s_t, a_t) + lpha \mathcal{H}(\pi(\cdot | s_t)) ight]$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class SquashedGaussianPolicy:
    """Outputs mean $\mu(s)$ and log standard deviation $\log \sigma(s)$ for tanh-squashed actions."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        self.state_dim = state_dim
        self.action_dim = action_dim

        std = np.sqrt(2.0 / state_dim)
        self.W1 = np.random.normal(0, std, (state_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)

        self.W_mu = np.random.normal(0, np.sqrt(2.0 / hidden_dim), (hidden_dim, action_dim))
        self.b_mu = np.zeros(action_dim)

        self.W_log_std = np.random.normal(0, np.sqrt(2.0 / hidden_dim), (hidden_dim, action_dim))
        self.b_log_std = np.zeros(action_dim)

    def forward(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = np.maximum(0, np.dot(state, self.W1) + self.b1)
        mu = np.dot(h, self.W_mu) + self.b_mu
        log_std = np.clip(np.dot(h, self.W_log_std) + self.b_log_std, -20.0, 2.0)
        return mu, log_std

    def sample(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        mu, log_std = self.forward(state)
        std = np.exp(log_std)
        noise = np.random.normal(0, 1, size=mu.shape)
        u = mu + std * noise
        a = np.tanh(u)  # Bound action to [-1, 1]

        # Enforce exact change-of-variables log-likelihood calculation
        log_prob = np.sum(-0.5 * (((u - mu) / std) ** 2 + 2.0 * log_std + np.log(2.0 * np.pi)))
        log_prob -= np.sum(np.log(np.clip(1.0 - a ** 2, 1e-6, 1.0)))
        return a, float(log_prob)


class SoftQNetwork:
    """Critic network estimating $Q(s, a)$."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        in_dim = state_dim + action_dim
        std = np.sqrt(2.0 / in_dim)
        self.W1 = np.random.normal(0, std, (in_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.normal(0, np.sqrt(2.0 / hidden_dim), (hidden_dim, 1))
        self.b2 = np.zeros(1)

    def forward(self, state: np.ndarray, action: np.ndarray) -> float:
        x = np.concatenate([state, action])
        h = np.maximum(0, np.dot(x, self.W1) + self.b1)
        q = float(np.dot(h, self.W2) + self.b2)
        return q
