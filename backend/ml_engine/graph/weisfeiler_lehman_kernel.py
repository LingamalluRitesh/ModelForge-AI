"""
ModelForge AI - ML Engine: Weisfeiler-Lehman Graph Isomorphism Kernel
Implements Shervashidze et al. Weisfeiler-Lehman Graph Kernels
measuring topological and structural similarity between graph pairs via iterative multiset subtree labeling.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import hashlib
import numpy as np


class WeisfeilerLehmanGraphKernel:
    """Computes Weisfeiler-Lehman subtree kernel similarity between graph structures."""
    def __init__(self, num_iterations: int = 3):
        self.h = num_iterations

    def _hash_label(self, label_str: str) -> str:
        return hashlib.md5(label_str.encode("utf-8")).hexdigest()[:8]

    def compute_similarity(
        self,
        adj1: np.ndarray,
        labels1: List[str],
        adj2: np.ndarray,
        labels2: List[str],
    ) -> float:
        """Calculate normalized kernel similarity in [0, 1]."""
        hist1: Dict[str, int] = {}
        hist2: Dict[str, int] = {}

        curr_l1 = list(labels1)
        curr_l2 = list(labels2)

        # Initialize base histograms
        for l in curr_l1:
            hist1[l] = hist1.get(l, 0) + 1
        for l in curr_l2:
            hist2[l] = hist2.get(l, 0) + 1

        # Iterative multiset neighborhood relabeling
        for it in range(self.h):
            next_l1 = []
            for i in range(len(curr_l1)):
                neighbors = np.where(adj1[i] > 0)[0]
                n_labels = sorted([curr_l1[n] for n in neighbors])
                multiset = f"{curr_l1[i]}_" + "_".join(n_labels)
                new_lbl = self._hash_label(multiset)
                next_l1.append(new_lbl)
                hist1[new_lbl] = hist1.get(new_lbl, 0) + 1

            next_l2 = []
            for i in range(len(curr_l2)):
                neighbors = np.where(adj2[i] > 0)[0]
                n_labels = sorted([curr_l2[n] for n in neighbors])
                multiset = f"{curr_l2[i]}_" + "_".join(n_labels)
                new_lbl = self._hash_label(multiset)
                next_l2.append(new_lbl)
                hist2[new_lbl] = hist2.get(new_lbl, 0) + 1

            curr_l1 = next_l1
            curr_l2 = next_l2

        # Dot product over shared feature counts
        all_keys = set(hist1.keys()).union(set(hist2.keys()))
        v1 = np.array([hist1.get(k, 0) for k in all_keys], dtype=float)
        v2 = np.array([hist2.get(k, 0) for k in all_keys], dtype=float)

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(v1, v2) / (norm1 * norm2))
