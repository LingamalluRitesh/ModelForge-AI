"""
ModelForge AI - ML Engine: Deep Deterministic Policy Gradient (DDPG)
Implements Lillicrap et al. Continuous Control with Deep Reinforcement Learning
with Deterministic Policy Gradient Theorem, Target Network Polyak Averaging, and Ornstein-Uhlenbeck Action Noise.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class OrnsteinUhlenbeckActionNoise:
    """Ornstein-Uhlenbeck exploration process for temporally correlated physical action dynamics."""
    def __init__(self, action_dim: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.full(action_dim, mu)

    def sample(self) -> np.ndarray:
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.normal(0, 1, self.action_dim)
        self.state += dx
        return self.state


class DDPGActor:
    """Deterministic Actor mapping continuous state to deterministic continuous action."""
    def __init__(self, state_dim: int, action_dim: int, max_action: float = 1.0):
        self.max_action = max_action
        std = np.sqrt(2.0 / state_dim)
        self.W1 = np.random.normal(0, std, (state_dim, 64))
        self.b1 = np.zeros(64)
        self.W2 = np.random.normal(0, np.sqrt(2.0 / 64), (64, action_dim))
        self.b2 = np.zeros(action_dim)

    def forward(self, state: np.ndarray) -> np.ndarray:
        h = np.maximum(0, np.dot(state, self.W1) + self.b1)
        a = self.max_action * np.tanh(np.dot(h, self.W2) + self.b2)
        return a
