"""
ModelForge AI - ML Engine: Multi-Agent Deep Deterministic Policy Gradient (MADDPG)
Implements Lowe et al. Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments
with Centralized Training with Decentralized Execution (CTDE) and deterministic policy gradients.
$
abla_{	heta_i} J(\mu_i) = \mathbb{E}_{\mathbf{x}, a \sim \mathcal{D}} \left[ 
abla_{	heta_i} \mu_i(a_i | o_i) 
abla_{a_i} Q_i^{\mu}(\mathbf{x}, a_1, \dots, a_N) |_{a_i = \mu_i(o_i)} ight]$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class CentralizedCritic:
    """Critic network observing joint state $x$ and all agents' actions $(a_1, \dots, a_N)$."""
    def __init__(self, total_state_dim: int, total_action_dim: int, hidden_dim: int = 128):
        in_dim = total_state_dim + total_action_dim
        std = np.sqrt(2.0 / in_dim)
        self.W1 = np.random.normal(0, std, (in_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.normal(0, np.sqrt(2.0 / hidden_dim), (hidden_dim, 1))
        self.b2 = np.zeros(1)

    def forward(self, all_states: np.ndarray, all_actions: np.ndarray) -> float:
        x = np.concatenate([all_states, all_actions])
        h = np.maximum(0, np.dot(x, self.W1) + self.b1)
        q = float(np.dot(h, self.W2) + self.b2)
        return q


class DecentralizedActor:
    """Actor network making decentralized action decisions based solely on local observation $o_i$."""
    def __init__(self, obs_dim: int, action_dim: int, max_action: float = 1.0):
        self.max_action = max_action
        std = np.sqrt(2.0 / obs_dim)
        self.W1 = np.random.normal(0, std, (obs_dim, 64))
        self.b1 = np.zeros(64)
        self.W2 = np.random.normal(0, np.sqrt(2.0 / 64), (64, action_dim))
        self.b2 = np.zeros(action_dim)

    def forward(self, local_obs: np.ndarray) -> np.ndarray:
        h = np.maximum(0, np.dot(local_obs, self.W1) + self.b1)
        a = self.max_action * np.tanh(np.dot(h, self.W2) + self.b2)
        return a
