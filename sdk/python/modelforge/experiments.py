"""
ModelForge AI SDK - Experiments, Training & AutoML Manager
"""

from typing import Any, Dict, List, Optional


class ExperimentRun:
    def __init__(self, data: Dict[str, Any]):
        self._data = data

    @property
    def id(self) -> str:
        return self._data.get("id", "")

    @property
    def name(self) -> str:
        return self._data.get("name", "")

    @property
    def algorithm(self) -> str:
        return self._data.get("algorithm_name", "")

    @property
    def status(self) -> str:
        return self._data.get("status", "")

    @property
    def metrics(self) -> Dict[str, Any]:
        return self._data.get("metrics", {})

    @property
    def hyperparameters(self) -> Dict[str, Any]:
        return self._data.get("hyperparameters", {})

    def __repr__(self) -> str:
        return f"<ModelForge ExperimentRun id='{self.id}' algo='{self.algorithm}' status='{self.status}'>"


class ExperimentManager:
    def __init__(self, client):
        self.client = client

    def list_runs(self, experiment_id: str) -> List[ExperimentRun]:
        res = self.client.request("GET", f"/experiments/{experiment_id}/runs")
        return [ExperimentRun(r) for r in res.get("data", [])]

    def train(
        self,
        project_id: str,
        name: str,
        algorithm: str,
        dataset_version_id: str,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        validation_split: float = 0.2,
    ) -> ExperimentRun:
        """Launch distributed model training run."""
        payload = {
            "name": name,
            "algorithm": algorithm,
            "dataset_version_id": dataset_version_id,
            "target_column": target_column,
            "feature_columns": feature_columns or [],
            "hyperparameters": hyperparameters or {},
            "validation_split": validation_split,
        }
        res = self.client.request("POST", f"/training/jobs?project_id={project_id}", json_data=payload)
        return ExperimentRun(res.get("data", {}))

    def run_automl(
        self,
        project_id: str,
        name: str,
        dataset_version_id: str,
        target_column: str,
        optimization_metric: str = "f1",
        max_trials: int = 20,
        time_budget_seconds: int = 1800,
    ) -> Dict[str, Any]:
        """Launch Bayesian AutoML Exploration with Optuna."""
        payload = {
            "name": name,
            "dataset_version_id": dataset_version_id,
            "target_column": target_column,
            "optimization_metric": optimization_metric,
            "max_trials": max_trials,
            "time_budget_seconds": time_budget_seconds,
        }
        res = self.client.request("POST", f"/automl/jobs?project_id={project_id}", json_data=payload)
        return res.get("data", {})
