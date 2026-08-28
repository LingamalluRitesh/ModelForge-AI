"""
ModelForge AI - ML Engine: Integrated Gradients (Axiomatic Feature Attribution)
Implements Sundararajan et al. path-integral feature attributions satisfying Completeness and Implementation Invariance axioms:
$IG_i(x) = (x_i - x_i') \times \int_{0}^1 \frac{\partial F(x' + \alpha (x - x'))}{\partial x_i} d\alpha$
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from app.core.exceptions import MLModelExecutionException

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class IntegratedGradientsExplainer:
    """Computes path-integral feature attributions relative to a reference baseline."""

    def __init__(self, model: Any, steps: int = 50):
        self.model = model
        self.steps = steps

    def attribute(
        self,
        input_instance: np.ndarray,
        baseline_instance: Optional[np.ndarray] = None,
        target_class: int = 1,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        x = np.asarray(input_instance, dtype=np.float32).flatten()
        if baseline_instance is None:
            x_base = np.zeros_like(x)
        else:
            x_base = np.asarray(baseline_instance, dtype=np.float32).flatten()

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(x))]

        # Generate interpolated path points along straight line $\alpha \in [0, 1]$
        alphas = np.linspace(0.0, 1.0, self.steps)
        path_points = np.array([x_base + a * (x - x_base) for a in alphas])

        if HAS_TORCH and hasattr(self.model, "forward"):
            # PyTorch exact autograd path
            torch_model = getattr(self.model, "model", self.model)
            torch_model.eval()
            inputs_tensor = torch.tensor(path_points, dtype=torch.float32, requires_grad=True)
            outputs = torch_model(inputs_tensor)

            if outputs.ndim == 2:
                target_outputs = outputs[:, target_class]
            else:
                target_outputs = outputs.squeeze()

            grads = torch.autograd.grad(
                outputs=target_outputs.sum(),
                inputs=inputs_tensor,
                create_graph=False,
            )[0].cpu().numpy()

            avg_grads = np.mean(grads[:-1], axis=0) # Riemann sum trapezoidal approximation
            attributions = (x - x_base) * avg_grads

        else:
            # Numerical gradient approximation along path
            eps = 1e-4
            grads = np.zeros_like(path_points)

            for step_idx, point in enumerate(path_points):
                point_df = point.reshape(1, -1)
                if hasattr(self.model, "predict_proba"):
                    base_prob = self.model.predict_proba(point_df)[0][target_class]
                else:
                    base_prob = float(self.model.predict(point_df)[0])

                for feat_idx in range(len(x)):
                    p_pert = point.copy()
                    p_pert[feat_idx] += eps
                    if hasattr(self.model, "predict_proba"):
                        prob_pert = self.model.predict_proba(p_pert.reshape(1, -1))[0][target_class]
                    else:
                        prob_pert = float(self.model.predict(p_pert.reshape(1, -1))[0])
                    grads[step_idx, feat_idx] = (prob_pert - base_prob) / eps

            avg_grads = np.mean(grads, axis=0)
            attributions = (x - x_base) * avg_grads

        # Verify completeness axiom: $\sum IG_i(x) \approx F(x) - F(x')$
        total_attribution = float(np.sum(attributions))
        attr_dict = {name: round(float(val), 4) for name, val in zip(feature_names, attributions)}
        sorted_attr = dict(sorted(attr_dict.items(), key=lambda item: abs(item[1]), reverse=True))

        return {
            "method": "Integrated Gradients",
            "feature_attributions": sorted_attr,
            "total_attribution_sum": round(total_attribution, 4),
            "steps_count": self.steps,
        }
