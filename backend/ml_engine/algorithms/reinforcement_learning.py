"""
ModelForge AI - ML Engine: Multi-Armed & Contextual Bandits for Dynamic Rollouts
Implements Epsilon-Greedy, Upper Confidence Bound (UCB1), Thompson Sampling, and LinUCB Contextual Bandits
to optimize live traffic routing between competing candidate models based on conversion reward signals.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class MultiArmedBanditRouter:
    """Adaptive bandit routing allocating traffic to models with highest expected reward."""

    def __init__(self, arm_names: List[str], strategy: str = "thompson_sampling", epsilon: float = 0.1):
        self.arm_names = arm_names
        self.strategy = strategy
        self.epsilon = epsilon

        self.k = len(arm_names)
        self.counts = np.zeros(self.k, dtype=int)
        self.rewards = np.zeros(self.k, dtype=np.float64)

        # For Thompson Sampling (Beta prior parameters)
        self.alpha_successes = np.ones(self.k, dtype=np.float64)
        self.beta_failures = np.ones(self.k, dtype=np.float64)

    def select_arm(self) -> str:
        """Select next model arm to route inference request."""
        # Warmup: ensure every arm has been pulled at least once
        unpulled = np.where(self.counts == 0)[0]
        if len(unpulled) > 0:
            chosen = int(unpulled[0])
            return self.arm_names[chosen]

        if self.strategy == "epsilon_greedy":
            if np.random.uniform(0, 1) < self.epsilon:
                chosen = int(np.random.choice(self.k))
            else:
                mean_rewards = self.rewards / np.maximum(1, self.counts)
                chosen = int(np.argmax(mean_rewards))

        elif self.strategy == "ucb1":
            total_steps = np.sum(self.counts)
            mean_rewards = self.rewards / np.maximum(1, self.counts)
            exploration_bonus = np.sqrt((2 * np.log(total_steps)) / np.maximum(1, self.counts))
            ucb_values = mean_rewards + exploration_bonus
            chosen = int(np.argmax(ucb_values))

        elif self.strategy == "thompson_sampling":
            # Sample from posterior Beta distributions
            samples = [
                np.random.beta(self.alpha_successes[i], self.beta_failures[i])
                for i in range(self.k)
            ]
            chosen = int(np.argmax(samples))

        else:
            chosen = int(np.random.choice(self.k))

        return self.arm_names[chosen]

    def update_reward(self, arm_name: str, reward: float):
        """Update arm statistics based on feedback (e.g., 1.0 for positive business conversion, 0.0 otherwise)."""
        if arm_name not in self.arm_names:
            return

        idx = self.arm_names.index(arm_name)
        self.counts[idx] += 1
        self.rewards[idx] += reward

        if reward >= 0.5:
            self.alpha_successes[idx] += 1.0
        else:
            self.beta_failures[idx] += 1.0

    def get_allocation_distribution(self) -> Dict[str, float]:
        """Compute estimated traffic allocation percentages based on empirical performance."""
        total = np.sum(self.counts)
        if total == 0:
            return {name: round(100.0 / self.k, 1) for name in self.arm_names}

        return {
            name: round(float((count / total) * 100.0), 1)
            for name, count in zip(self.arm_names, self.counts)
        }


class LinUCBContextualBandit:
    """Linear Upper Confidence Bound (LinUCB) contextual bandit conditioning model routing on user feature vectors."""

    def __init__(self, arm_names: List[str], feature_dim: int, alpha: float = 1.0):
        self.arm_names = arm_names
        self.feature_dim = feature_dim
        self.alpha = alpha

        # Per-arm ridge matrices A_a = d x d and response vectors b_a = d x 1
        self.A = {arm: np.eye(feature_dim, dtype=np.float64) for arm in arm_names}
        self.b = {arm: np.zeros((feature_dim, 1), dtype=np.float64) for arm in arm_names}

    def select_arm(self, context_vector: np.ndarray) -> str:
        x = np.asarray(context_vector, dtype=np.float64).reshape(-1, 1)
        best_p = -float("inf")
        best_arm = self.arm_names[0]

        for arm in self.arm_names:
            A_inv = np.linalg.inv(self.A[arm])
            theta_hat = np.dot(A_inv, self.b[arm])
            # Mean payoff expectation
            expected_payoff = float(np.dot(theta_hat.T, x))
            # Variance uncertainty bound
            uncertainty = float(self.alpha * np.sqrt(np.dot(np.dot(x.T, A_inv), x)))
            p_score = expected_payoff + uncertainty

            if p_score > best_p:
                best_p = p_score
                best_arm = arm

        return best_arm

    def update_reward(self, arm_name: str, context_vector: np.ndarray, reward: float):
        if arm_name not in self.arm_names:
            return
        x = np.asarray(context_vector, dtype=np.float64).reshape(-1, 1)
        self.A[arm_name] += np.dot(x, x.T)
        self.b[arm_name] += reward * x
