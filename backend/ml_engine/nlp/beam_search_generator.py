"""
ModelForge AI - NLP Engine: Constrained Beam Search with Length Penalty & N-Gram Blocking
Implements Graves Sequence Transduction with Recurrent Neural Networks Beam Search
with Wu et al. Google NMT Length Penalty $lpha=0.6$ and 3-gram repetition blocking.
$	ext{score}(Y, X) = rac{\log P(Y | X)}{rac{(5 + |Y|)^lpha}{(5 + 1)^lpha}}$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import heapq
import numpy as np


class BeamHypothesis:
    def __init__(self, tokens: List[int], log_prob: float):
        self.tokens = tokens
        self.log_prob = log_prob

    def length_penalized_score(self, alpha: float = 0.6) -> float:
        L = len(self.tokens)
        penalty = ((5.0 + L) ** alpha) / ((5.0 + 1.0) ** alpha)
        return self.log_prob / max(1e-5, penalty)

    def __lt__(self, other: "BeamHypothesis") -> bool:
        return self.length_penalized_score() < other.length_penalized_score()


class ConstrainedBeamSearch:
    """Maintains Top-K beam hypotheses with strict n-gram blocking."""
    def __init__(self, beam_width: int = 5, length_penalty_alpha: float = 0.6, no_repeat_ngram_size: int = 3):
        self.k = beam_width
        self.alpha = length_penalty_alpha
        self.ngram_block = no_repeat_ngram_size

    def search(
        self,
        initial_token: int,
        predict_next_logits_fn: Callable[[List[int]], np.ndarray],
        max_steps: int = 30,
        eos_token_id: int = 2,
    ) -> List[int]:
        beams = [BeamHypothesis([initial_token], 0.0)]

        for _ in range(max_steps):
            candidates: List[BeamHypothesis] = []

            for hyp in beams:
                if hyp.tokens[-1] == eos_token_id:
                    candidates.append(hyp)
                    continue

                logits = predict_next_logits_fn(hyp.tokens)
                # Log softmax
                shift = logits - np.max(logits)
                log_probs = shift - np.log(np.sum(np.exp(shift)))

                # Top-2K token expansions
                top_tokens = np.argsort(log_probs)[- (2 * self.k) :]
                for tok in top_tokens:
                    tok = int(tok)
                    # Check n-gram repetition blocking
                    if len(hyp.tokens) >= self.ngram_block - 1:
                        ngram = tuple(hyp.tokens[- (self.ngram_block - 1) :] + [tok])
                        # Prevent duplicate n-grams
                        existing_ngrams = [
                            tuple(hyp.tokens[i : i + self.ngram_block])
                            for i in range(len(hyp.tokens) - self.ngram_block + 1)
                        ]
                        if ngram in existing_ngrams:
                            continue

                    candidates.append(BeamHypothesis(hyp.tokens + [tok], hyp.log_prob + float(log_probs[tok])))

            # Retain top-K beams
            candidates.sort(key=lambda item: item.length_penalized_score(self.alpha), reverse=True)
            beams = candidates[: self.k]

            # If all top beams terminated at EOS
            if all(b.tokens[-1] == eos_token_id for b in beams):
                break

        return beams[0].tokens
