"""
ModelForge AI - NLP Engine: Top-p (Nucleus) Sampling & Temperature Scaling
Implements Holtzman et al. The Curious Case of Neural Text Degeneration
truncating token probability distribution to the smallest set whose cumulative mass exceeds threshold $p$.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class NucleusSampler:
    """Samples from top-p cumulative probability nucleus."""
    def __init__(self, top_p: float = 0.90, temperature: float = 0.7, repetition_penalty: float = 1.1):
        self.p = top_p
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty

    def sample_next_token(self, logits: np.ndarray, generated_tokens: List[int]) -> int:
        logits = np.array(logits, dtype=float)

        # 1. Apply repetition penalty
        for tok in set(generated_tokens):
            if logits[tok] < 0:
                logits[tok] *= self.repetition_penalty
            else:
                logits[tok] /= self.repetition_penalty

        # 2. Temperature scaling
        scaled_logits = logits / max(1e-4, self.temperature)
        shift = scaled_logits - np.max(scaled_logits)
        probs = np.exp(shift) / np.sum(np.exp(shift))

        # 3. Top-p Nucleus Truncation
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)

        # Remove tokens with cumulative probability above threshold p
        cutoff_idx = np.searchsorted(cumulative_probs, self.p)
        top_indices = sorted_indices[: cutoff_idx + 1]
        top_probs = probs[top_indices]
        top_probs /= np.sum(top_probs)

        # 4. Sample token from filtered distribution
        sampled_token = np.random.choice(top_indices, p=top_probs)
        return int(sampled_token)
