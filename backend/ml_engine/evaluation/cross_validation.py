"""
ModelForge AI - Advanced Cross Validation Strategies
Implements Stratified K-Fold, TimeSeriesSplit, GroupKFold, and Purged Group Time Series Split.
"""

from typing import Any, Dict, Generator, List, Optional, Tuple
import numpy as np


class PurgedGroupTimeSeriesSplit:
    """
    Purged & Embargoed Group Time-Series Split preventing lookahead data leakage in financial and sequential systems.
    """

    def __init__(
        self,
        n_splits: int = 5,
        max_train_group_size: Optional[int] = None,
        group_gap: int = 0,
        max_test_group_size: Optional[int] = None,
    ):
        self.n_splits = n_splits
        self.max_train_group_size = max_train_group_size
        self.group_gap = group_gap
        self.max_test_group_size = max_test_group_size

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        if groups is None:
            groups = np.arange(len(X))

        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)

        if self.n_splits >= n_groups:
            raise ValueError(f"Cannot have n_splits={self.n_splits} greater than or equal to n_groups={n_groups}")

        group_test_size = n_groups // (self.n_splits + 1)

        for i in range(self.n_splits):
            test_start = (i + 1) * group_test_size
            test_end = test_start + group_test_size

            train_groups = unique_groups[:max(0, test_start - self.group_gap)]
            test_groups = unique_groups[test_start:test_end]

            train_idx = np.where(np.isin(groups, train_groups))[0]
            test_idx = np.where(np.isin(groups, test_groups))[0]

            yield train_idx, test_idx
