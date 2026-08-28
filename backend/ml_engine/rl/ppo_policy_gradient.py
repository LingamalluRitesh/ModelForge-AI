"""
ModelForge AI - ML Engine: Proximal Policy Optimization (PPO)
Implements Schulman et al. Proximal Policy Optimization with Clipped Surrogate Objective
and Generalized Advantage Estimation (GAE-$\lambda$) for continuous & discrete policy control.
$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right) \right]$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class ActorCriticNetwork:
    """Shared feature representation with separate Policy (Actor) and Value (Critic) heads."""

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 64):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

        std = np.sqrt(2.0 / state_dim)
        # Shared trunk
        self.W_trunk = np.random.normal(0, std, (state_dim, hidden_dim))
        self.b_trunk = np.zeros(hidden_dim)

        # Actor head: logits over discrete actions
        self.W_actor = np.random.normal(0, np.sqrt(2.0 / hidden_dim), (hidden_dim, n_actions))
        self.b_actor = np.zeros(n_actions)

        # Critic head: scalar state-value V(s)
        self.W_critic = np.random.normal(0, np.sqrt(2.0 / hidden_dim), (hidden_dim, 1))
        self.b_critic = np.zeros(1)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        shift_x = x - np.max(x, axis=-1, keepdims=True)
        exps = np.exp(shift_x)
        return exps / np.sum(exps, axis=-1, keepdims=True)

    def forward(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns action probabilities $(B, n\_actions)$ and state value $V(s)$ $(B, 1)$."""
        h = np.tanh(np.dot(state, self.W_trunk) + self.b_trunk)
        logits = np.dot(h, self.W_actor) + self.b_actor
        probs = self._softmax(logits)
        value = np.dot(h, self.W_critic) + self.b_critic
        return probs, value


class PPOAgent:
    """Proximal Policy Optimization Agent with GAE advantage calculation."""

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        lr: float = 0.003,
        epochs_per_update: int = 4,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.lr = lr
        self.epochs_per_update = epochs_per_update

        self.ac = ActorCriticNetwork(state_dim, n_actions)

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generalized Advantage Estimation:
        $\delta_t^V = r_t + \gamma V(s_{t+1}) (1 - d_t) - V(s_t)$
        $\hat{A}_t = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V$
        """
        T = len(rewards)
        advantages = np.zeros(T)
        last_gae_lam = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            non_terminal = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * next_val * non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * non_terminal * last_gae_lam

        returns = advantages + values
        # Normalize advantages
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages) + 1e-8
        norm_advantages = (advantages - adv_mean) / adv_std

        return norm_advantages, returns

    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Sample action from stochastic policy and return action, log_prob, and value."""
        state_t = np.asarray(state, dtype=np.float64).reshape(1, -1)
        probs, value = self.ac.forward(state_t)
        probs = probs[0]
        action = int(np.random.choice(len(probs), p=probs))
        log_prob = float(np.log(max(1e-10, probs[action])))
        return action, log_prob, float(value[0, 0])
