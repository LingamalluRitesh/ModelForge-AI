"""
ModelForge AI - RL Engine: Continuous Action Proximal Policy Optimization (PPO)
Implements Schulman et al. Proximal Policy Optimization Algorithms with Clipped Surrogate Objective,
Generalized Advantage Estimation (GAE-Lambda), and Gaussian Policy Heads.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class ContinuousPPOActor:
    """Gaussian Policy Network outputting mean and standard deviation vectors."""
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim

        std = np.sqrt(2.0 / state_dim)
        self.W1 = np.random.normal(0, std, (state_dim, 64))
        self.b1 = np.zeros(64)
        self.W_mu = np.random.normal(0, np.sqrt(2.0 / 64), (64, action_dim))
        self.b_mu = np.zeros(action_dim)
        self.log_std = np.zeros(action_dim)

    def forward(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = np.maximum(0, np.dot(state, self.W1) + self.b1)
        mu = np.tanh(np.dot(h, self.W_mu) + self.b_mu)
        std = np.exp(self.log_std)
        return mu, std

    def sample_action(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        mu, std = self.forward(state)
        action = np.random.normal(mu, std)
        # Log probability of Gaussian
        log_prob = -0.5 * np.sum(((action - mu) / std) ** 2 + 2.0 * self.log_std + np.log(2.0 * np.pi))
        return action, float(log_prob)
