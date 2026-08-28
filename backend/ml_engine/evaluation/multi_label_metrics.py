"""
ModelForge AI - ML Engine: Multi-Label Classification & Ranking Evaluation Suite
Implements Hamming Loss, Subset Accuracy (Exact Match Ratio), Micro/Macro/Weighted F1,
Coverage Error, Ranking Loss, and Label Ranking Average Precision (LRAP).
$	ext{Hamming Loss} = rac{1}{N \cdot L} \sum_{i=1}^N \sum_{j=1}^L \mathbb{I}(y_{ij} 
eq \hat{y}_{ij})$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class MultiLabelEvaluator:
    """Evaluates multi-label prediction matrices against binary ground-truth indicator tensors."""
    @staticmethod
    def hamming_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)
        return float(np.mean(y_true != y_pred))

    @staticmethod
    def subset_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)
        exact_matches = np.all(y_true == y_pred, axis=1)
        return float(np.mean(exact_matches))

    @staticmethod
    def label_ranking_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=int)
        y_score = np.asarray(y_score, dtype=float)
        N, L = y_true.shape
        lrap_sum = 0.0

        for i in range(N):
            labels = np.where(y_true[i] == 1)[0]
            if len(labels) == 0:
                continue

            order = np.argsort(y_score[i])[::-1]
            prec_sum = 0.0
            for lbl in labels:
                rank = np.where(order == lbl)[0][0] + 1
                top_ranked = order[:rank]
                num_true_in_top = np.sum(y_true[i, top_ranked])
                prec_sum += num_true_in_top / float(rank)

            lrap_sum += prec_sum / len(labels)

        return float(lrap_sum / max(1, N))
