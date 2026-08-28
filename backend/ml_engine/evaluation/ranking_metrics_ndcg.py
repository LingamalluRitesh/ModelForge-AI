"""
ModelForge AI - ML Engine: Information Retrieval & Ranking Metrics Suite
Implements Normalized Discounted Cumulative Gain (NDCG@K), Mean Reciprocal Rank (MRR),
and Mean Average Precision (MAP@K) for recommendation and ranking evaluations.
$	ext{DCG}@K = \sum_{i=1}^K rac{2^{rel_i} - 1}{\log_2(i + 1)}$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class RankingMetricsEvaluator:
    """Computes standard Information Retrieval evaluation metrics."""
    @staticmethod
    def dcg_at_k(relevance_scores: np.ndarray, k: int = 10) -> float:
        rel = np.asarray(relevance_scores, dtype=float)[:k]
        if len(rel) == 0:
            return 0.0
        gains = 2.0 ** rel - 1.0
        discounts = np.log2(np.arange(2, len(rel) + 2))
        return float(np.sum(gains / discounts))

    @staticmethod
    def ndcg_at_k(relevance_scores: np.ndarray, k: int = 10) -> float:
        dcg = RankingMetricsEvaluator.dcg_at_k(relevance_scores, k)
        ideal_scores = np.sort(relevance_scores)[::-1]
        idcg = RankingMetricsEvaluator.dcg_at_k(ideal_scores, k)
        if idcg == 0.0:
            return 0.0
        return float(dcg / idcg)

    @staticmethod
    def mean_reciprocal_rank(recommendation_lists: List[List[int]], ground_truth_items: List[int]) -> float:
        rr_sum = 0.0
        for recs, target in zip(recommendation_lists, ground_truth_items):
            if target in recs:
                rank = recs.index(target) + 1
                rr_sum += 1.0 / rank
        return float(rr_sum / max(1, len(ground_truth_items)))
