"""
ModelForge AI - Preprocessing: Statistical TF-IDF & Okapi BM25 Text Feature Engine
Extracts word and character n-grams with Sublinear Term Frequency scaling and BM25 document ranking.
$	ext{BM25}(D, Q) = \sum_{i=1}^n 	ext{IDF}(q_i) \cdot rac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot rac{|D|}{	ext{avgdl}}ight)}$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math
import numpy as np


class BM25Ranker:
    """Okapi BM25 document ranking and feature score generator."""
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_lens_: List[int] = []
        self.avgdl_: float = 0.0
        self.idf_: Dict[str, float] = {}
        self.corpus_size_: int = 0

    def fit(self, documents: List[str]):
        self.corpus_size_ = len(documents)
        doc_term_freqs = []
        total_len = 0
        df_counts: Dict[str, int] = {}

        for doc in documents:
            words = doc.lower().split()
            total_len += len(words)
            self.doc_lens_.append(len(words))

            unique_terms = set(words)
            for t in unique_terms:
                df_counts[t] = df_counts.get(t, 0) + 1

        self.avgdl_ = total_len / max(1, self.corpus_size_)
        # Robertson-Spärck Jones IDF
        for term, df in df_counts.items():
            self.idf_[term] = math.log((self.corpus_size_ - df + 0.5) / (df + 0.5) + 1.0)

        return self

    def score(self, query: str, doc_tokens: List[str]) -> float:
        words = query.lower().split()
        doc_len = len(doc_tokens)
        score = 0.0

        for w in words:
            if w in self.idf_:
                tf = doc_tokens.count(w)
                idf = self.idf_[w]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1e-5, self.avgdl_)))
                score += idf * (numerator / denominator)

        return float(score)
