import pytest
from backend.app.services.bayesian_hpo_service import BayesianOptimizationEngine


def test_hpo_initial_exploration_and_reporting():
    engine = BayesianOptimizationEngine()
    p1 = engine.suggest_parameters()
    assert "learning_rate" in p1
    assert "batch_size" in p1
    assert 1e-5 <= p1["learning_rate"] <= 1e-1

    engine.report_trial(p1, loss=0.45, accuracy=0.88)
    p2 = engine.suggest_parameters()
    engine.report_trial(p2, loss=0.32, accuracy=0.93)
    p3 = engine.suggest_parameters()
    engine.report_trial(p3, loss=0.55, accuracy=0.82)

    best = engine.get_best_hyperparameters()
    assert best is not None
    assert best["loss"] == 0.32
    assert best["accuracy"] == 0.93


def test_hpo_exploitation_phase():
    engine = BayesianOptimizationEngine()
    for i in range(5):
        params = engine.suggest_parameters()
        engine.report_trial(params, loss=0.5 - (i * 0.05), accuracy=0.8 + (i * 0.03))

    suggested = engine.suggest_parameters()
    assert len(suggested) == 4
    assert 0.0 <= suggested["dropout"] <= 0.5
