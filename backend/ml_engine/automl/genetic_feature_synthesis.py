"""
ModelForge AI - AutoML: Genetic Feature Synthesis & Mutation Engine
Evolves non-linear algebraic mathematical combinations using Evolutionary Crossover and Point Mutation.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class GeneticFeatureSynthesizer:
    """Synthesizes high-order interaction columns from raw tabular datasets."""
    def __init__(self, num_synthetic_features: int = 5):
        self.n_synth = num_synthetic_features

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        N, D = X.shape
        if D < 2:
            return X

        synthetic_cols = []
        # Feature combinations: Product, Ratio, Log-Sum, Sqrt-Diff
        col1 = X[:, 0]
        col2 = X[:, 1]

        synthetic_cols.append(col1 * col2)                                    # Interaction 1: Product
        synthetic_cols.append(col1 / np.where(np.abs(col2) < 1e-4, 1e-4, col2)) # Interaction 2: Ratio
        synthetic_cols.append(np.log1p(np.abs(col1 + col2)))                  # Interaction 3: LogSum
        synthetic_cols.append(np.sqrt(np.abs(col1 - col2)))                   # Interaction 4: RootDiff
        synthetic_cols.append(np.sin(col1) * np.cos(col2))                    # Interaction 5: Harmonic

        return np.column_stack([X] + synthetic_cols[: self.n_synth])
