import pytest
from backend.app.services.experiment_run_exporter import ExperimentRunExporter


def test_experiment_run_lifecycle_and_export():
    exporter = ExperimentRunExporter()
    run = exporter.create_run(
        experiment_id="exp_tabnet_v1",
        run_name="batch_size_64_lr_1e3",
        parameters={"learning_rate": 0.001, "batch_size": 64},
        tags={"dataset": "credit_risk_v2"},
    )
    assert run["run_id"].startswith("run_")
    assert run["status"] == "RUNNING"

    exporter.log_metric(run["run_id"], "loss", 0.45, step=1)
    exporter.log_metric(run["run_id"], "loss", 0.32, step=2)
    exporter.log_metric(run["run_id"], "accuracy", 0.91, step=2)

    ended = exporter.end_run(run["run_id"], status="FINISHED")
    assert ended["status"] == "FINISHED"

    mlflow_data = exporter.export_mlflow_format(run["run_id"])
    assert mlflow_data["run_info"]["status"] == "FINISHED"
    assert mlflow_data["data"]["metrics"]["loss"] == 0.32
    assert mlflow_data["data"]["metrics"]["accuracy"] == 0.91
    assert mlflow_data["data"]["params"]["batch_size"] == 64
