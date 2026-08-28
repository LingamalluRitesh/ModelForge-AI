"""
ModelForge AI - ML Engine: Twin Delayed Deep Deterministic Policy Gradients (TD3)
Implements Fujimoto et al. Addressing Function Approximation Error in Actor-Critic Methods
with Clipped Double Q-Learning, Delayed Policy Updates, and Target Policy Smoothing.
$y = r + \gamma \min_{i=1,2} Q_{	heta'_i}(s', 	ext{clip}(\pi_{\phi'}(s') + \epsilon, a_{low}, a_{high}))$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class TD3Actor:
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


class TD3Critic:
    """Twin Q-Networks Q1 and Q2 to prevent overestimation bias."""
    def __init__(self, state_dim: int, action_dim: int):
        in_dim = state_dim + action_dim
        std = np.sqrt(2.0 / in_dim)
        # Q1
        self.W1 = np.random.normal(0, std, (in_dim, 64))
        self.b1 = np.zeros(64)
        self.W1_out = np.random.normal(0, np.sqrt(2.0 / 64), (64, 1))
        self.b1_out = np.zeros(1)

        # Q2
        self.W2 = np.random.normal(0, std, (in_dim, 64))
        self.b2 = np.zeros(64)
        self.W2_out = np.random.normal(0, np.sqrt(2.0 / 64), (64, 1))
        self.b2_out = np.zeros(1)

    def forward(self, state: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = np.concatenate([state, action], axis=-1)
        h1 = np.maximum(0, np.dot(x, self.W1) + self.b1)
        q1 = np.dot(h1, self.W1_out) + self.b1_out

        h2 = np.maximum(0, np.dot(x, self.W2) + self.b2)
        q2 = np.dot(h2, self.W2_out) + self.b2_out
        return q1, q2
