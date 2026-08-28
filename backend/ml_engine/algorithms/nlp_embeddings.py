"""
ModelForge AI - Text Processing & Feature Embeddings
Implements TF-IDF, Count Vectorizer, and n-gram sub-word extractors for hybrid tabular-text modeling.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class AdvancedTextFeatureExtractor:
    """
    Extracts numerical representation from unstructured textual columns.
    """

    def __init__(
        self,
        max_features: int = 100,
        ngram_range: tuple = (1, 2),
        use_idf: bool = True,
        stop_words: str = "english",
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.use_idf = use_idf
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            use_idf=use_idf,
            stop_words=stop_words,
        )
        self.is_fitted = False

    def fit(self, texts: List[str]):
        self.vectorizer.fit(texts)
        self.is_fitted = True
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Extractor is not fitted.")
        return self.vectorizer.transform(texts).toarray()

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)

    @property
    def feature_names(self) -> List[str]:
        return list(self.vectorizer.get_feature_names_out())
