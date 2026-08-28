"""
ModelForge AI - AutoML: Hyperband Bandit-Based Configuration Evaluation
Implements Li, Jamieson, DeSalvo, Rostamizadeh, & Talwalkar Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization
using dynamic resource allocation and Successive Halving brackets.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math
import numpy as np


class HyperbandScheduler:
    """Dynamic resource bandit scheduler executing Successive Halving."""
    def __init__(self, max_resource_epochs: int = 81, eta: int = 3):
        self.R = max_resource_epochs
        self.eta = eta
        self.s_max = int(math.floor(math.log(self.R) / math.log(eta)))
        self.B = (self.s_max + 1) * self.R

    def get_brackets_plan(self) -> List[Dict[str, Any]]:
        brackets = []
        for s in reversed(range(self.s_max + 1)):
            # Initial number of configurations
            n = int(math.ceil((self.B / self.R) * (self.eta ** s) / (s + 1)))
            # Initial resource allocated per config
            r = self.R * (self.eta ** (-s))

            brackets.append({
                "bracket_index": s,
                "num_configs": n,
                "initial_resource": int(r),
                "num_halving_rounds": s + 1,
            })
        return brackets
