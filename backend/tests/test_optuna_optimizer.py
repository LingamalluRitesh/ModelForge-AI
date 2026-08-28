"""
ModelForge AI - Optuna Bayesian HPO Unit Tests
"""

import numpy as np
import pytest
from ml_engine.optimization.optuna_optimizer import OptunaHPOEngine


def test_optuna_hpo_optimization():
    np.random.seed(42)
    X = np.random.randn(120, 6)
    y = ((X[:, 0] * 2 + X[:, 1]) > 0).astype(int)

    optimizer = OptunaHPOEngine(
        algorithm_name="random_forest",
        problem_type="classification",
        n_trials=3,
        optimization_metric="accuracy",
        timeout_seconds=30,
    )
    result = optimizer.optimize(X, y)

    assert "best_score" in result
    assert "best_params" in result
    assert "trials_history" in result
    assert result["best_score"] > 0.50
    assert result["trials_count"] == 3
