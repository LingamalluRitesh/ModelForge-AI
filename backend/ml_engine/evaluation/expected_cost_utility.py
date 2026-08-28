"""
ModelForge AI - Evaluation: Bayesian Expected Utility & Value-at-Risk (VaR)
Computes financial portfolio value-at-risk, conditional value-at-risk (CVaR / Expected Shortfall),
and von Neumann-Morgenstern expected utility across calibrated model predictions.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class PortfolioRiskEvaluator:
    """Financial Value-at-Risk (VaR) and Expected Shortfall (CVaR) calculator."""
    @staticmethod
    def value_at_risk(returns_or_losses: np.ndarray, confidence_level: float = 0.95) -> Dict[str, float]:
        losses = np.asarray(returns_or_losses, dtype=float)
        # VaR at alpha level
        var_threshold = float(np.percentile(losses, confidence_level * 100))
        # Conditional Value at Risk (Expected Shortfall): Mean of losses exceeding VaR
        tail_losses = losses[losses >= var_threshold]
        cvar = float(np.mean(tail_losses)) if len(tail_losses) > 0 else var_threshold

        return {
            "confidence_level": confidence_level,
            "value_at_risk_var": round(var_threshold, 4),
            "expected_shortfall_cvar": round(cvar, 4),
            "worst_case_drawdown": round(float(np.max(losses)), 4),
        }
