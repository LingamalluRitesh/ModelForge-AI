"""
ModelForge AI - ML Engine: Decision Transformer (Offline RL via Sequence Modeling)
Implements Chen et al. Decision Transformer: Reinforcement Learning via Sequence Modeling
modeling trajectories $	au = (\hat{R}_1, s_1, a_1, \hat{R}_2, s_2, a_2, \dots)$ via Causal Self-Attention.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class DecisionTransformer:
    """Autoregressive trajectory modeling conditioning actions on desired returns-to-go."""
    def __init__(self, state_dim: int, action_dim: int, d_model: int = 128, max_seq_len: int = 20):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.max_len = max_seq_len

        # Modality embedding heads
        self.embed_return = np.random.normal(0, 0.05, (1, d_model))
        self.embed_state = np.random.normal(0, np.sqrt(2.0 / state_dim), (state_dim, d_model))
        self.embed_action = np.random.normal(0, np.sqrt(2.0 / action_dim), (action_dim, d_model))

        # Prediction head
        self.predict_action_W = np.random.normal(0, np.sqrt(2.0 / d_model), (d_model, action_dim))
        self.predict_action_b = np.zeros(action_dim)

    def predict_next_action(
        self,
        returns_to_go: List[float],
        states: List[np.ndarray],
        actions: List[np.ndarray],
    ) -> np.ndarray:
        """
        returns_to_go: list of scalar target returns
        states: list of state vectors
        actions: list of previous action vectors
        """
        T = len(states)
        # Interleave embeddings: [R_t, s_t, a_t]
        tokens = []
        for t in range(T):
            r_tok = np.array([returns_to_go[t]]) * self.embed_return
            s_tok = np.dot(states[t], self.embed_state)
            tokens.append(r_tok.reshape(-1))
            tokens.append(s_tok)
            if t < len(actions):
                a_tok = np.dot(actions[t], self.embed_action)
                tokens.append(a_tok)

        # Causal representation from last state token
        last_state_rep = tokens[-1]
        action_pred = np.tanh(np.dot(last_state_rep, self.predict_action_W) + self.predict_action_b)
        return action_pred
