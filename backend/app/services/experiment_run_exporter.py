"""
MLflow & W&B Compatible Experiment Run Exporter.
Serializes experiment parameters, metric step histories, artifact URIs, and run lineage.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import json
import uuid


class ExperimentRunExporter:
    """Manages experiment run tracking and serialization in standard MLOps exchange formats."""

    def __init__(self):
        self.runs: Dict[str, Dict[str, Any]] = {}

    def create_run(
        self,
        experiment_id: str,
        run_name: str,
        parameters: Dict[str, Any] = None,
        tags: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()

        run = {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "run_name": run_name,
            "status": "RUNNING",
            "start_time": now,
            "end_time": None,
            "parameters": parameters or {},
            "metrics_history": {},
            "tags": tags or {"framework": "pytorch", "user": "mlops_engineer"},
            "artifacts": [],
        }
        self.runs[run_id] = run
        return run

    def log_metric(self, run_id: str, key: str, value: float, step: int) -> None:
        if run_id not in self.runs:
            raise KeyError(f"Run {run_id} does not exist.")

        if key not in self.runs[run_id]["metrics_history"]:
            self.runs[run_id]["metrics_history"][key] = []

        self.runs[run_id]["metrics_history"][key].append({
            "step": step,
            "value": value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def end_run(self, run_id: str, status: str = "FINISHED") -> Dict[str, Any]:
        if run_id not in self.runs:
            raise KeyError(f"Run {run_id} does not exist.")

        self.runs[run_id]["status"] = status
        self.runs[run_id]["end_time"] = datetime.now(timezone.utc).isoformat()
        return self.runs[run_id]

    def export_mlflow_format(self, run_id: str) -> Dict[str, Any]:
        if run_id not in self.runs:
            raise KeyError(f"Run {run_id} does not exist.")

        r = self.runs[run_id]
        latest_metrics = {
            k: v[-1]["value"] for k, v in r["metrics_history"].items() if v
        }

        return {
            "run_info": {
                "run_uuid": r["run_id"],
                "experiment_id": r["experiment_id"],
                "status": r["status"],
                "start_time": r["start_time"],
                "end_time": r["end_time"],
            },
            "data": {
                "metrics": latest_metrics,
                "params": r["parameters"],
                "tags": r["tags"],
            },
        }
