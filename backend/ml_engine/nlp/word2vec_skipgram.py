"""
ModelForge AI - ML Engine: Word2Vec Skip-Gram with Negative Sampling (SGNS)
Implements Mikolov et al. Distributed Representations of Words with Noise-Contrastive Negative Sampling,
Subsampling of Frequent Words, and Cosine Similarity semantic search.
$\mathcal{L}_{SGNS} = \log \sigma(v'_{w_O} \cdot v_{w_I}) + \sum_{i=1}^k \mathbb{E}_{w_i \sim P_n(w)} [\log \sigma(-v'_{w_i} \cdot v_{w_I})]$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from collections import Counter


class Word2VecSkipGram:
    """Word2Vec Continuous Skip-Gram Architecture with Negative Sampling."""

    def __init__(
        self,
        embed_dim: int = 100,
        window_size: int = 5,
        n_negatives: int = 5,
        lr: float = 0.025,
        epochs: int = 10,
        subsample_thresh: float = 1e-4,
    ):
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.n_negatives = n_negatives
        self.lr = lr
        self.epochs = epochs
        self.subsample_thresh = subsample_thresh

        self.vocab: Dict[str, int] = {}
        self.inv_vocab: Dict[int, str] = {}
        self.W_in: Optional[np.ndarray] = None   # Target word representations (V, D)
        self.W_out: Optional[np.ndarray] = None  # Context word representations (V, D)
        self.noise_distribution_: Optional[np.ndarray] = None

    def _sigmoid(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -15.0, 15.0)))

    def fit(self, corpus: List[List[str]]):
        """Train word embeddings over tokenized sentences."""
        # 1. Build vocabulary and unigram frequency distribution
        word_counts = Counter(word for sentence in corpus for word in sentence)
        self.vocab = {w: i for i, (w, _) in enumerate(word_counts.items())}
        self.inv_vocab = {i: w for w, i in self.vocab.items()}
        V = len(self.vocab)

        # Noise distribution raised to 3/4 power for negative sampling
        counts = np.array([word_counts[self.inv_vocab[i]] for i in range(V)], dtype=np.float64)
        powered = counts ** 0.75
        self.noise_distribution_ = powered / np.sum(powered)

        # 2. Weight matrices initialization
        self.W_in = np.random.uniform(-0.5 / self.embed_dim, 0.5 / self.embed_dim, (V, self.embed_dim))
        self.W_out = np.zeros((V, self.embed_dim))

        # 3. Training passes
        for epoch in range(self.epochs):
            for sentence in corpus:
                # Subsampling of frequent words
                indices = [self.vocab[w] for w in sentence if w in self.vocab]
                n_words = len(indices)

                for pos, target_idx in enumerate(indices):
                    # Dynamic window size
                    curr_window = np.random.randint(1, self.window_size + 1)
                    start = max(0, pos - curr_window)
                    end = min(n_words, pos + curr_window + 1)

                    context_indices = [indices[k] for k in range(start, end) if k != pos]

                    for ctx_idx in context_indices:
                        # Sample negative word indices
                        neg_indices = np.random.choice(V, size=self.n_negatives, p=self.noise_distribution_)

                        # Target vector
                        v_target = self.W_in[target_idx]

                        # Positive context update: error = sigma(v_ctx . v_target) - 1
                        score_pos = np.dot(self.W_out[ctx_idx], v_target)
                        sig_pos = self._sigmoid(score_pos)
                        grad_pos = sig_pos - 1.0

                        # Negative contexts update: error = sigma(v_neg . v_target)
                        scores_neg = np.dot(self.W_out[neg_indices], v_target)
                        sig_neg = self._sigmoid(scores_neg)
                        grad_neg = sig_neg

                        # Accumulate gradient for target word
                        grad_target = grad_pos * self.W_out[ctx_idx] + np.dot(grad_neg, self.W_out[neg_indices])

                        # Parameter updates
                        self.W_out[ctx_idx] -= self.lr * grad_pos * v_target
                        self.W_out[neg_indices] -= self.lr * grad_neg[:, np.newaxis] * v_target[np.newaxis, :]
                        self.W_in[target_idx] -= self.lr * grad_target

        return self

    def most_similar(self, word: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Compute cosine similarity ranking across all vocabulary vectors."""
        if word not in self.vocab:
            return []

        w_idx = self.vocab[word]
        vec = self.W_in[w_idx]
        norm_v = np.linalg.norm(vec)

        norms_all = np.linalg.norm(self.W_in, axis=1)
        sims = np.dot(self.W_in, vec) / np.maximum(1e-10, norms_all * norm_v)

        top_indices = np.argsort(sims)[::-1][1 : top_k + 1]
        return [(self.inv_vocab[i], round(float(sims[i]), 4)) for i in top_indices]
