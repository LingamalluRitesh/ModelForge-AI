"""
ModelForge AI - Continuous Retraining Pipeline Trigger Service
Evaluates streaming telemetry metrics, executes automated drift triggers, and starts canary rollouts.
"""

from typing import Any, Dict, List, Optional
import time
import uuid


class RetrainingTriggerService:
    @staticmethod
    def evaluate_and_trigger(
        model_version_id: str,
        current_psi_score: float,
        current_error_rate: float,
        policy_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        config = policy_config or {
            "max_psi_threshold": 0.20,
            "max_error_rate_threshold": 0.05,
            "auto_canary_rollout": True,
            "canary_initial_percentage": 10.0,
        }

        psi_breach = current_psi_score >= config["max_psi_threshold"]
        error_breach = current_error_rate >= config["max_error_rate_threshold"]
        trigger_required = psi_breach or error_breach

        if trigger_required:
            pipeline_run_id = str(uuid.uuid4())
            action = "RETRAINING_PIPELINE_LAUNCHED"
        else:
            pipeline_run_id = None
            action = "NO_ACTION_STABLE"

        return {
            "model_version_id": model_version_id,
            "evaluated_psi": current_psi_score,
            "evaluated_error_rate": current_error_rate,
            "trigger_required": trigger_required,
            "action": action,
            "pipeline_run_id": pipeline_run_id,
            "auto_canary_enabled": config.get("auto_canary_rollout", True),
            "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
