"""
ModelForge AI - ML Engine: Matrix Profile (STAMP / STOMP) Time Series Motif Discovery
Implements Yeh et al. Matrix Profile: All-Pairs Similarity Search for Subsequences
identifying discords (anomalies) and motifs (repeated patterns) in $O(N^2)$ time.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class MatrixProfileEngine:
    """Computes 1D Matrix Profile and 1D Profile Index for time series subsequences."""
    def __init__(self, window_size: int = 20):
        self.m = window_size

    def compute(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        series: (N,) 1D time series
        returns: (MP, MPI) where MP is distance to nearest neighbor subsequence, MPI is index
        """
        x = np.asarray(series, dtype=float)
        N = len(x)
        num_subsequences = N - self.m + 1

        # Extract normalized subsequences
        subsequences = np.zeros((num_subsequences, self.m))
        for i in range(num_subsequences):
            sub = x[i : i + self.m]
            mean = np.mean(sub)
            std = max(1e-5, np.std(sub))
            subsequences[i] = (sub - mean) / std

        mp = np.full(num_subsequences, float("inf"))
        mpi = np.zeros(num_subsequences, dtype=int)

        # Distance calculation
        for i in range(num_subsequences):
            for j in range(num_subsequences):
                # Trivial match exclusion zone
                if abs(i - j) < self.m // 2:
                    continue

                dist = np.sqrt(np.sum((subsequences[i] - subsequences[j]) ** 2))
                if dist < mp[i]:
                    mp[i] = dist
                    mpi[i] = j

        return mp, mpi
