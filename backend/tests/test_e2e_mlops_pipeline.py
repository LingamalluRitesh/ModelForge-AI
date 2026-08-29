"""
End-to-End MLOps Pipeline & Model Promotion Stress Suite.
Tests HPO Tuning, Experiment Run Serialization, Artifact Bytecode Security, and Model Registry Promotion.
"""

import pickle
import pytest
from backend.app.services.bayesian_hpo_service import BayesianOptimizationEngine
from backend.app.services.experiment_run_exporter import ExperimentRunExporter
from backend.app.services.artifact_integrity_service import ArtifactIntegrityService


def test_full_mlops_lifecycle_pipeline():
    # 1. Hyperparameter Optimization Search
    hpo = BayesianOptimizationEngine()
    for step in range(5):
        params = hpo.suggest_parameters()
        loss = 0.50 - (step * 0.08)
        acc = 0.80 + (step * 0.03)
        hpo.report_trial(params, loss=loss, accuracy=acc)

    best_trial = hpo.get_best_hyperparameters()
    assert best_trial is not None
    assert best_trial["loss"] < 0.25

    # 2. Experiment Run Metric Logging
    exporter = ExperimentRunExporter()
    run = exporter.create_run(
        experiment_id="exp_resnet_tabular_v1",
        run_name="resnet_best_hpo",
        parameters=best_trial["params"],
    )
    for epoch in range(10):
        exporter.log_metric(run["run_id"], "val_loss", 0.40 - (epoch * 0.03), step=epoch)
        exporter.log_metric(run["run_id"], "val_auc", 0.85 + (epoch * 0.01), step=epoch)

    completed_run = exporter.end_run(run["run_id"], status="FINISHED")
    assert completed_run["status"] == "FINISHED"
    assert len(completed_run["metrics_history"]["val_loss"]) == 10

    # 3. Artifact Packaging & Bytecode Security
    security = ArtifactIntegrityService()
    weights_payload = pickle.dumps({
        "model_architecture": "ResNetTabular",
        "state_dict": {"layer1.weight": [0.1, 0.2], "layer2.weight": [0.3, 0.4]},
        "hyperparameters": best_trial["params"],
    })

    scan_result = security.scan_pickle_bytecode(weights_payload)
    assert scan_result["is_safe_for_loading"] is True

    fingerprint = security.compute_artifact_fingerprint(weights_payload)
    is_valid = security.verify_artifact_signature(weights_payload, fingerprint["hmac_signature"])
    assert is_valid is True
