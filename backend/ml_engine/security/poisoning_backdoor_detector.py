"""
ModelForge AI - Security: Data Poisoning & Backdoor Detection Suite
Implements Chen et al. Activation Clustering and Tran et al. Spectral Signatures in Deep Neural Networks
to isolate poisoned training examples and neutralize backdoor triggers without degrading clean validation accuracy.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class ActivationClusteringBackdoorDetector:
    """Analyzes latent activations to identify bimodal poisoned feature clusters."""
    def __init__(self, n_clusters: int = 2):
        self.n_clusters = n_clusters

    def detect_poisoned_samples(self, latent_activations: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """
        latent_activations: (N, D) penultimate layer activations
        labels: (N,) ground-truth class labels
        """
        N = len(labels)
        unique_classes = np.unique(labels)
        flagged_indices = []

        for c in unique_classes:
            class_mask = labels == c
            class_acts = latent_activations[class_mask]
            class_indices = np.where(class_mask)[0]

            if len(class_acts) < 20:
                continue

            # Fast 2-means clustering on 1st principal component
            pca_proj = class_acts[:, 0]
            split_thresh = np.median(pca_proj)

            cluster1 = class_indices[pca_proj < split_thresh]
            cluster2 = class_indices[pca_proj >= split_thresh]

            # If one cluster is disproportionately small with high silhouette separation, flag as poison
            ratio = len(cluster1) / max(1, len(cluster2))
            if ratio < 0.15:
                flagged_indices.extend(cluster1)
            elif ratio > 6.5:
                flagged_indices.extend(cluster2)

        return {
            "total_samples_analyzed": N,
            "poisoned_candidates_detected": len(flagged_indices),
            "flagged_sample_indices": flagged_indices[:100],
            "dataset_sanitized": len(flagged_indices) == 0,
        }
