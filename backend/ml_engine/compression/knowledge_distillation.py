"""
ModelForge AI - ML Engine: Knowledge Distillation & Model Compression
Implements Hinton et al. Distilling the Knowledge in a Neural Network with Temperature-Scaled Soft Targets,
Kullback-Leibler Divergence Loss, and Teacher-Student Feature Hints Matching.
$\mathcal{L}_{KD} = (1 - lpha) \mathcal{L}_{CE}(y, \sigma(z_s)) + lpha T^2 \mathcal{L}_{KL}\left(\sigma\left(rac{z_t}{T}ight), \sigma\left(rac{z_s}{T}ight)ight)$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class KnowledgeDistiller:
    """Teacher-Student Knowledge Distillation Engine."""
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        self.temperature = temperature
        self.alpha = alpha

    def _softmax(self, logits: np.ndarray, T: float) -> np.ndarray:
        scaled = logits / T
        shift = scaled - np.max(scaled, axis=-1, keepdims=True)
        exps = np.exp(shift)
        return exps / np.sum(exps, axis=-1, keepdims=True)

    def loss(
        self,
        student_logits: np.ndarray,
        teacher_logits: np.ndarray,
        true_labels: np.ndarray,
    ) -> float:
        """Calculate weighted cross-entropy and soft distillation loss."""
        # 1. Hard Cross-Entropy Loss
        p_student = self._softmax(student_logits, T=1.0)
        n_samples = len(true_labels)
        ce_loss = -np.mean(np.log(np.clip(p_student[np.arange(n_samples), true_labels], 1e-10, 1.0)))

        # 2. Soft KL Divergence Loss
        p_soft_student = self._softmax(student_logits, T=self.temperature)
        p_soft_teacher = self._softmax(teacher_logits, T=self.temperature)

        kl_div = np.sum(
            p_soft_teacher * (np.log(np.clip(p_soft_teacher, 1e-10, 1.0)) - np.log(np.clip(p_soft_student, 1e-10, 1.0))),
            axis=-1,
        )
        kl_loss = np.mean(kl_div)

        total_loss = (1.0 - self.alpha) * ce_loss + self.alpha * (self.temperature ** 2) * kl_loss
        return float(total_loss)
