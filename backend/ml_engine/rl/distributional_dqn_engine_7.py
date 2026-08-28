"""
ModelForge AI - RL Engine: Categorical Distributional Q-Learning Engine 7
Implements Bellemare, Dabney, & Munos A Distributional Perspective on Reinforcement Learning (C51).
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class DistributionalQNetwork_7:
    """C51 Distributional Value Network outputting discrete probability distributions across N atoms."""

    def __init__(self, state_dim: int, action_dim: int, num_atoms: int = 51, v_min: float = -10.0, v_max: float = 10.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.support = np.linspace(v_min, v_max, num_atoms)
        self.delta_z = (v_max - v_min) / float(num_atoms - 1)

        std = np.sqrt(2.0 / state_dim)
        self.W1 = np.random.normal(0, std, (state_dim, 128))
        self.b1 = np.zeros(128)
        self.W2 = np.random.normal(0, np.sqrt(2.0 / 128), (128, action_dim * num_atoms))
        self.b2 = np.zeros(action_dim * num_atoms)

    def forward(self, state: np.ndarray) -> np.ndarray:
        h = np.maximum(0, np.dot(state, self.W1) + self.b1)
        logits = (np.dot(h, self.W2) + self.b2).reshape(self.action_dim, self.num_atoms)
        shift = logits - np.max(logits, axis=-1, keepdims=True)
        probs = np.exp(shift) / np.sum(np.exp(shift), axis=-1, keepdims=True)
        return probs

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        probs = self.forward(state)
        return np.dot(probs, self.support)
