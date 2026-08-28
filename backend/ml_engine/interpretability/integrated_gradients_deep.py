"""
ModelForge AI - ML Engine: Path-Integrated Gradients
Implements Sundararajan, Taly, & Yan Axiomatic Attribution for Deep Networks (Integrated Gradients)
satisfying Completeness, Implementation Invariance, Linearity, and Sensitivity axioms.
$	ext{IG}_i(x) = (x_i - x_i') 	imes \int_{0}^1 rac{\partial F(x' + lpha (x - x'))}{\partial x_i} dlpha$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class IntegratedGradientsEngine:
    """Path-integrated gradient attribution engine with Riemann sum approximation."""
    def __init__(self, steps: int = 50):
        self.steps = steps

    def attribute(
        self,
        grad_fn: Callable[[np.ndarray], np.ndarray],
        instance: np.ndarray,
        baseline: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        x = np.asarray(instance, dtype=float)
        x_base = np.zeros_like(x) if baseline is None else np.asarray(baseline, dtype=float)
        D = len(x)

        if feature_names is None:
            feature_names = [f"x_{i}" for i in range(D)]

        # Generate linear interpolation path: x' + alpha * (x - x')
        alphas = np.linspace(0.0, 1.0, self.steps)
        path_samples = np.array([x_base + a * (x - x_base) for a in alphas])

        # Evaluate gradients along path
        grads = np.array([grad_fn(sample) for sample in path_samples])

        # Riemann trapezoidal integration
        avg_grads = np.mean(grads, axis=0)
        attributions = (x - x_base) * avg_grads

        # Construct feature ranking
        results = []
        for i, name in enumerate(feature_names):
            results.append({
                "feature": name,
                "attribution": round(float(attributions[i]), 5),
                "input_value": round(float(x[i]), 4),
                "baseline_value": round(float(x_base[i]), 4),
            })

        results.sort(key=lambda item: abs(item["attribution"]), reverse=True)

        return {
            "total_attribution": round(float(np.sum(attributions)), 5),
            "steps": self.steps,
            "attributions": results,
        }
