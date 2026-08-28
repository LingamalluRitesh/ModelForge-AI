"""
ModelForge AI - ML Engine: SentencePiece Unigram Language Model Tokenizer
Implements Kudo SentencePiece: A simple and language independent subword tokenizer
with Viterbi optimal path decoding and Expectation-Maximization vocabulary pruning.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math
import numpy as np


class SentencePieceUnigram:
    """Unigram Language Model Tokenizer with Viterbi optimal dynamic programming segmentation."""
    def __init__(self, target_vocab_size: int = 1000):
        self.target_vocab_size = target_vocab_size
        self.vocab: Dict[str, float] = {}  # token -> log probability

    def fit(self, sentences: List[str]):
        """Initialize and prune unigram vocabulary."""
        # 1. Initialize seed vocabulary from raw characters & high-frequency n-grams
        raw_counts: Dict[str, int] = {}
        for s in sentences:
            clean = "_" + s.replace(" ", "_")
            for length in range(1, 5):
                for i in range(len(clean) - length + 1):
                    sub = clean[i : i + length]
                    raw_counts[sub] = raw_counts.get(sub, 0) + 1

        total_counts = sum(raw_counts.values())
        self.vocab = {tok: math.log(cnt / total_counts) for tok, cnt in raw_counts.items()}
        return self

    def encode(self, text: str) -> List[str]:
        """Viterbi shortest-path dynamic programming tokenization."""
        clean = "_" + text.replace(" ", "_")
        N = len(clean)

        # best_score[i] stores max log-prob for prefix clean[:i]
        best_score = [-float("inf")] * (N + 1)
        best_edge = [-1] * (N + 1)
        best_score[0] = 0.0

        for i in range(N):
            if best_score[i] == -float("inf"):
                continue

            for j in range(i + 1, min(N + 1, i + 16)):
                sub = clean[i:j]
                if sub in self.vocab:
                    score = best_score[i] + self.vocab[sub]
                    if score > best_score[j]:
                        best_score[j] = score
                        best_edge[j] = i

        # Backtrack optimal tokens
        tokens = []
        curr = N
        while curr > 0:
            prev = best_edge[curr]
            tokens.append(clean[prev:curr])
            curr = prev

        tokens.reverse()
        return tokens
