"""
ModelForge AI - ML Engine: Hindsight Experience Replay (HER)
Implements Andrychowicz et al. Hindsight Experience Replay
transforming failed rollouts into successful demonstrations by retroactively re-substituting visited states as goals.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class HindsightExperienceReplayBuffer:
    """HER Buffer with 'future' goal re-labeling strategy."""
    def __init__(self, capacity: int = 50000, k_replays: int = 4):
        self.capacity = capacity
        self.k = k_replays
        self.buffer: List[Dict[str, Any]] = []

    def store_episode(self, episode_transitions: List[Dict[str, Any]]):
        T = len(episode_transitions)
        for t, trans in enumerate(episode_transitions):
            # Store original transition
            self.buffer.append(trans)

            # Sample k future states to create virtual successful hindsight goals
            for _ in range(self.k):
                future_idx = np.random.randint(t, T)
                hindsight_goal = episode_transitions[future_idx]["next_state"]

                # Relabel transition with virtual goal
                virtual_trans = dict(trans)
                virtual_trans["goal"] = hindsight_goal
                # Compute surrogate reward: 0 if at goal, -1 otherwise
                at_goal = np.linalg.norm(trans["next_state"] - hindsight_goal) < 0.05
                virtual_trans["reward"] = 0.0 if at_goal else -1.0

                self.buffer.append(virtual_trans)

        if len(self.buffer) > self.capacity:
            self.buffer = self.buffer[-self.capacity :]
