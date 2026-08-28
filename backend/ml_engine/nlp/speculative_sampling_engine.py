"""
ModelForge AI - NLP Engine: Speculative Decoding & Verification Engine
Implements Leviathan et al. Fast Inference from Transformers via Speculative Decoding
and Chen et al. Accelerating Large Language Model Decoding with Speculative Sampling
using small draft models to propose $\gamma$ candidate tokens verified in parallel by a target LLM.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class SpeculativeDecoder:
    """Accelerates autoregressive sequence generation via draft propose-and-verify loops."""
    def __init__(
        self,
        draft_model_fn: Callable[[List[int]], np.ndarray],
        target_model_fn: Callable[[List[int]], np.ndarray],
        gamma_lookahead: int = 4,
        temperature: float = 0.7,
    ):
        self.draft_fn = draft_model_fn
        self.target_fn = target_model_fn
        self.gamma = gamma_lookahead
        self.temperature = temperature

    def _sample_from_logits(self, logits: np.ndarray) -> int:
        scaled = logits / max(1e-4, self.temperature)
        shift = scaled - np.max(scaled)
        probs = np.exp(shift) / np.sum(np.exp(shift))
        return int(np.random.choice(len(probs), p=probs))

    def generate_step(self, prefix: List[int]) -> List[int]:
        # 1. Draft Phase: autoregressively sample gamma tokens
        draft_tokens = []
        curr_prefix = list(prefix)
        draft_probs_list = []

        for _ in range(self.gamma):
            logits = self.draft_fn(curr_prefix)
            scaled = logits / max(1e-4, self.temperature)
            shift = scaled - np.max(scaled)
            probs = np.exp(shift) / np.sum(np.exp(shift))
            tok = int(np.random.choice(len(probs), p=probs))
            draft_tokens.append(tok)
            draft_probs_list.append(probs)
            curr_prefix.append(tok)

        # 2. Target Verification Phase: parallel evaluation of prefix + draft_tokens
        accepted_tokens = []
        for i, tok in enumerate(draft_tokens):
            eval_prefix = prefix + accepted_tokens
            target_logits = self.target_fn(eval_prefix)
            scaled = target_logits / max(1e-4, self.temperature)
            shift = scaled - np.max(scaled)
            target_probs = np.exp(shift) / np.sum(np.exp(shift))

            p_target = target_probs[tok]
            p_draft = draft_probs_list[i][tok]

            # Rejection sampling criterion: accept with prob min(1, p_target / p_draft)
            accept_prob = min(1.0, p_target / max(1e-8, p_draft))
            if np.random.rand() < accept_prob:
                accepted_tokens.append(tok)
            else:
                # Resample corrected token from max(0, p_target - p_draft)
                residual_dist = np.maximum(0.0, target_probs - draft_probs_list[i])
                if np.sum(residual_dist) > 0:
                    residual_dist /= np.sum(residual_dist)
                    corrected_tok = int(np.random.choice(len(residual_dist), p=residual_dist))
                else:
                    corrected_tok = int(np.argmax(target_probs))
                accepted_tokens.append(corrected_tok)
                break

        return accepted_tokens
