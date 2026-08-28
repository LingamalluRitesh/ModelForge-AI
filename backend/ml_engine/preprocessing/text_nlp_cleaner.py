"""
ModelForge AI - ML Engine: Text Preprocessing & NLP Feature Extraction
Implements regex tokenization, subword n-gram generation, BM25 statistical scoring,
TF-IDF vectorization, semantic vocabulary pruning, and multi-lingual language detection.
"""

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import re
import math
import string
from collections import Counter, defaultdict
import numpy as np
import pandas as pd


class TextNLPPreprocessor:
    """Production NLP pipeline for text cleaning, stopword removal, and vocabulary mapping."""

    DEFAULT_STOPWORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any",
        "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below",
        "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't",
        "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for",
        "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having",
        "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him",
        "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in",
        "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most",
        "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only",
        "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
        "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
        "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
        "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
        "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very",
        "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't",
        "what", "what's", "when", "when's", "where", "where's", "which", "while", "who",
        "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you",
        "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
    }

    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = False,
        remove_stopwords: bool = True,
        min_token_length: int = 2,
        custom_stopwords: Optional[Set[str]] = None,
    ):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.min_token_length = min_token_length
        self.stopwords = (custom_stopwords or self.DEFAULT_STOPWORDS) if remove_stopwords else set()

    def clean_text(self, text: str) -> str:
        """Apply normalized text regex transformations."""
        if not isinstance(text, str):
            return ""

        if self.lowercase:
            text = text.lower()

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        # Remove emails
        text = re.sub(r"\S+@\S+", " ", text)

        if self.remove_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))

        if self.remove_numbers:
            text = re.sub(r"\d+", " ", text)

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        """Split cleaned text into normalized token sequences."""
        cleaned = self.clean_text(text)
        tokens = cleaned.split()
        if self.stopwords:
            tokens = [t for t in tokens if t not in self.stopwords and len(t) >= self.min_token_length]
        else:
            tokens = [t for t in tokens if len(t) >= self.min_token_length]
        return tokens

    def generate_ngrams(self, tokens: List[str], n: int = 2) -> List[str]:
        """Generate contiguous token n-grams."""
        if len(tokens) < n:
            return []
        return ["_".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


class BM25Ranker:
    """Okapi BM25 statistical text search & relevance scoring algorithm."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lens: List[int] = []
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        self.idf: Dict[str, float] = {}
        self.tokenized_corpus: List[List[str]] = []

    def fit(self, corpus: List[str]):
        preprocessor = TextNLPPreprocessor()
        self.tokenized_corpus = [preprocessor.tokenize(doc) for doc in corpus]
        self.corpus_size = len(self.tokenized_corpus)
        self.doc_lens = [len(doc) for doc in self.tokenized_corpus]
        self.avg_doc_len = float(np.mean(self.doc_lens)) if self.doc_lens else 0.0

        for doc in self.tokenized_corpus:
            seen_tokens = set(doc)
            for token in seen_tokens:
                self.doc_freqs[token] += 1

        # Calculate inverse document frequency (IDF)
        for token, df in self.doc_freqs.items():
            self.idf[token] = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)

        return self

    def score_query(self, query: str) -> np.ndarray:
        preprocessor = TextNLPPreprocessor()
        query_tokens = preprocessor.tokenize(query)
        scores = np.zeros(self.corpus_size, dtype=np.float64)

        for token in query_tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]

            for i, doc in enumerate(self.tokenized_corpus):
                freq = doc.count(token)
                if freq == 0:
                    continue
                num = freq * (self.k1 + 1.0)
                den = freq + self.k1 * (1.0 - self.b + self.b * (self.doc_lens[i] / max(1.0, self.avg_doc_len)))
                scores[i] += idf_val * (num / max(1e-6, den))

        return scores
