"""
ModelForge AI - ML Engine: Asynchronous Advantage Actor-Critic (A3C)
Implements Mnih et al. Asynchronous Methods for Deep Reinforcement Learning
with Asynchronous Gradient Updates, Generalized Advantage Estimation, and Shared Global Parameter Server.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class A3CWorker:
    """Asynchronous Actor-Critic Worker collecting trajectory rollouts."""
    def __init__(self, state_dim: int, action_dim: int, gamma: float = 0.99):
        self.gamma = gamma
        self.state_dim = state_dim
        self.action_dim = action_dim

        std = np.sqrt(2.0 / state_dim)
        self.W_shared = np.random.normal(0, std, (state_dim, 64))
        self.b_shared = np.zeros(64)

        # Policy head (Actor)
        self.W_policy = np.random.normal(0, np.sqrt(2.0 / 64), (64, action_dim))
        self.b_policy = np.zeros(action_dim)

        # Value head (Critic)
        self.W_value = np.random.normal(0, np.sqrt(2.0 / 64), (64, 1))
        self.b_value = np.zeros(1)

    def forward(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        h = np.maximum(0, np.dot(state, self.W_shared) + self.b_shared)
        # Softmax policy
        logits = np.dot(h, self.W_policy) + self.b_policy
        shift = logits - np.max(logits)
        probs = np.exp(shift) / np.sum(np.exp(shift))

        # State value
        value = float(np.dot(h, self.W_value) + self.b_value)
        return probs, value

    def compute_discounted_returns(self, rewards: List[float], next_value: float) -> List[float]:
        R = next_value
        returns = []
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return returns
