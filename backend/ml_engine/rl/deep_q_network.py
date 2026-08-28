"""
ModelForge AI - ML Engine: Deep Q-Networks (DQN) & Reinforcement Learning Suite
Implements Mnih et al. Deep Q-Networks with Double Q-Learning, Dueling Architecture heads,
and Prioritized Experience Replay (PER) with Importance Sampling (IS) weights.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class PrioritizedReplayBuffer:
    """Proportional Prioritized Experience Replay (PER) with SumTree structure."""

    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta_start: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self.priorities: np.ndarray = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        experience = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience

        self.priorities[self.pos] = self.max_priority ** self.alpha
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[List[Any], np.ndarray, np.ndarray]:
        n = len(self.buffer)
        probs = self.priorities[:n] / np.sum(self.priorities[:n])
        indices = np.random.choice(n, batch_size, p=probs)

        # Importance sampling weights: w_i = (N * P(i))^(-beta) / max_w
        weights = (n * probs[indices]) ** (-self.beta)
        weights = weights / np.max(weights)

        batch = [self.buffer[idx] for idx in indices]
        return batch, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        for idx, error in zip(indices, td_errors):
            p = (abs(error) + 1e-5) ** self.alpha
            self.priorities[idx] = p
            self.max_priority = max(self.max_priority, abs(error) + 1e-5)


class DuelingDQNHead:
    """Wang et al. Dueling Network separating State Value $V(s)$ and Action Advantage $A(s, a)$."""

    def __init__(self, in_features: int, n_actions: int):
        self.in_features = in_features
        self.n_actions = n_actions

        std = np.sqrt(2.0 / in_features)
        # Value stream: V(s) -> 1
        self.W_val = np.random.normal(0, std, (in_features, 1))
        self.b_val = np.zeros(1)

        # Advantage stream: A(s, a) -> n_actions
        self.W_adv = np.random.normal(0, std, (in_features, n_actions))
        self.b_adv = np.zeros(n_actions)

    def forward(self, features: np.ndarray) -> np.ndarray:
        """$Q(s, a) = V(s) + \left(A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a')\right)$."""
        V = np.dot(features, self.W_val) + self.b_val  # (B, 1)
        A = np.dot(features, self.W_adv) + self.b_adv  # (B, n_actions)

        Q = V + (A - np.mean(A, axis=-1, keepdims=True))
        return Q


class DeepQAgent:
    """Double Dueling Deep Q-Network Agent."""

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        gamma: float = 0.99,
        lr: float = 0.001,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.online_net = DuelingDQNHead(state_dim, n_actions)
        self.target_net = DuelingDQNHead(state_dim, n_actions)
        self.replay_buffer = PrioritizedReplayBuffer()

    def select_action(self, state: np.ndarray, evaluation: bool = False) -> int:
        if not evaluation and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)

        state_t = np.asarray(state, dtype=np.float64).reshape(1, -1)
        q_vals = self.online_net.forward(state_t)
        return int(np.argmax(q_vals[0]))

    def train_step(self, batch_size: int = 64) -> float:
        if len(self.replay_buffer.buffer) < batch_size:
            return 0.0

        batch, indices, is_weights = self.replay_buffer.sample(batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])

        # Current Q-values
        curr_q = self.online_net.forward(states)
        curr_q_actions = curr_q[np.arange(batch_size), actions]

        # Double DQN Target: Q_target(s', argmax_a Q_online(s', a))
        next_q_online = self.online_net.forward(next_states)
        best_next_actions = np.argmax(next_q_online, axis=-1)

        next_q_target = self.target_net.forward(next_states)
        target_q_vals = next_q_target[np.arange(batch_size), best_next_actions]

        # Bellman equation: r + gamma * Q_target * (1 - done)
        targets = rewards + self.gamma * target_q_vals * (1.0 - dones.astype(float))

        td_errors = targets - curr_q_actions
        self.replay_buffer.update_priorities(indices, td_errors)

        # Decay exploration
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        loss = float(np.mean(is_weights * (td_errors ** 2)))
        return loss
