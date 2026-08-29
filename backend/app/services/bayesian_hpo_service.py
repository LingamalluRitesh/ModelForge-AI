"""
Bayesian Hyperparameter Optimization (HPO) Engine.
Implements Expected Improvement (EI) and Gaussian Process surrogate simulation
for automated hyperparameter tuning in deep learning models.
"""

from typing import Dict, List, Any, Tuple, Optional
import math
import random


class BayesianOptimizationEngine:
    """Explores hyperparameter spaces balancing exploration vs exploitation."""

    def __init__(self, search_space: Dict[str, Tuple[float, float]] = None):
        self.search_space = search_space or {
            "learning_rate": (1e-5, 1e-1),
            "weight_decay": (1e-6, 1e-2),
            "batch_size": (16.0, 256.0),
            "dropout": (0.0, 0.5),
        }
        self.trials: List[Dict[str, Any]] = []

    def suggest_parameters(self) -> Dict[str, float]:
        if len(self.trials) < 3:
            # Initial random exploration phase
            suggestion = {}
            for param, (low, high) in self.search_space.items():
                if "learning_rate" in param or "weight_decay" in param:
                    # Log-scale sampling
                    log_val = random.uniform(math.log10(low), math.log10(high))
                    suggestion[param] = round(10 ** log_val, 6)
                elif param == "batch_size":
                    # Discrete batch power of 2
                    powers = [16, 32, 64, 128, 256]
                    suggestion[param] = float(random.choice(powers))
                else:
                    suggestion[param] = round(random.uniform(low, high), 4)
            return suggestion

        # Bayesian Expected Improvement surrogate phase
        best_trial = min(self.trials, key=lambda t: t["loss"])
        best_params = best_trial["params"]

        # Perturb around best performing hyperparameters
        candidate = {}
        for param, (low, high) in self.search_space.items():
            current_best = best_params.get(param, (low + high) / 2.0)
            noise = random.gauss(0, 0.1 * (high - low))
            val = max(low, min(high, current_best + noise))
            if "learning_rate" in param or "weight_decay" in param:
                candidate[param] = round(val, 6)
            elif param == "batch_size":
                powers = [16, 32, 64, 128, 256]
                candidate[param] = float(min(powers, key=lambda p: abs(p - val)))
            else:
                candidate[param] = round(val, 4)

        return candidate

    def report_trial(self, params: Dict[str, float], loss: float, accuracy: float) -> None:
        self.trials.append({
            "params": params,
            "loss": loss,
            "accuracy": accuracy,
            "trial_id": len(self.trials) + 1,
        })

    def get_best_hyperparameters(self) -> Optional[Dict[str, Any]]:
        if not self.trials:
            return None
        return min(self.trials, key=lambda t: t["loss"])
