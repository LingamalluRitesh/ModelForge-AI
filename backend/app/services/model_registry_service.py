"""
ModelForge AI - Model Registry & Governance Domain Service
Handles Model Registration, Versioning, Quality Gates Evaluation, and Approval Workflows.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.exceptions import EntityNotFoundException, ValidationException, AuthorizationException
from app.models.model_registry import (
    RegisteredModel, ModelVersion, ModelApprovalRequest, QualityGateConfig, ModelStage
)
from app.models.experiment import ExperimentRun
from app.schemas.model_registry import (
    RegisteredModelCreate, ModelVersionCreate, ModelApprovalRequestCreate, ModelApprovalVote
)


class ModelRegistryService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def list_models(self, project_id: str) -> List[RegisteredModel]:
        query = select(RegisteredModel).where(RegisteredModel.project_id == project_id, RegisteredModel.is_archived == False)
        res = await self.session.execute(query)
        return list(res.scalars().all())

    async def create_registered_model(self, project_id: str, user_id: str, payload: RegisteredModelCreate) -> RegisteredModel:
        model = RegisteredModel(
            project_id=project_id,
            name=payload.name,
            description=payload.description,
            problem_type=payload.problem_type,
            tags=payload.tags or [],
            created_by_id=user_id,
        )
        self.session.add(model)
        await self.session.commit()
        return model

    async def register_model_version(
        self,
        registered_model_id: str,
        user_id: str,
        payload: ModelVersionCreate,
    ) -> ModelVersion:
        """Create new version under registered model from an experiment run."""
        query = select(ExperimentRun).where(ExperimentRun.id == payload.experiment_run_id)
        res = await self.session.execute(query)
        run = res.scalar_one_or_none()
        if not run:
            raise EntityNotFoundException("ExperimentRun", payload.experiment_run_id)

        if not run.model_artifact_uri:
            raise ValidationException("Experiment run does not have a saved model artifact.")

        # Check quality gates
        gate_summary = self._evaluate_quality_gate(run.metrics)

        version = ModelVersion(
            registered_model_id=registered_model_id,
            version_tag=payload.version_tag,
            stage=ModelStage.DEVELOPMENT,
            experiment_run_id=run.id,
            algorithm_name=run.algorithm_name,
            framework=run.framework,
            storage_uri=run.model_artifact_uri,
            metrics=run.metrics or {},
            hyperparameters=run.hyperparameters or {},
            quality_gate_passed=gate_summary["passed"],
            quality_gate_summary=gate_summary,
            description=payload.description,
            created_by_id=user_id,
        )
        self.session.add(version)
        await self.session.commit()
        return version

    async def request_approval(self, version_id: str, user_id: str, payload: ModelApprovalRequestCreate) -> ModelApprovalRequest:
        """Submit model version for governance approval before promoting to Staging or Production."""
        query = select(ModelVersion).where(ModelVersion.id == version_id)
        res = await self.session.execute(query)
        version = res.scalar_one_or_none()
        if not version:
            raise EntityNotFoundException("ModelVersion", version_id)

        # Transition stage to validation / candidate
        version.stage = ModelStage.CANDIDATE

        approval = ModelApprovalRequest(
            model_version_id=version_id,
            target_stage=payload.target_stage,
            status="pending",
            requested_by_id=user_id,
            quality_checks_passed=version.quality_gate_passed,
            review_comments=payload.notes,
        )
        self.session.add(approval)
        await self.session.commit()
        return approval

    async def review_approval(self, approval_id: str, reviewer_id: str, payload: ModelApprovalVote) -> ModelApprovalRequest:
        """Reviewer approves or rejects the promotion request."""
        query = select(ModelApprovalRequest).where(ModelApprovalRequest.id == approval_id)
        res = await self.session.execute(query)
        approval = res.scalar_one_or_none()
        if not approval:
            raise EntityNotFoundException("ModelApprovalRequest", approval_id)

        approval.status = "approved" if payload.action == "approve" else "rejected"
        approval.reviewed_by_id = reviewer_id
        approval.reviewed_at = datetime.now(timezone.utc)
        approval.review_comments = payload.review_comments

        # Update model version stage
        query = select(ModelVersion).where(ModelVersion.id == approval.model_version_id)
        res = await self.session.execute(query)
        version = res.scalar_one_or_none()

        if version:
            if payload.action == "approve":
                version.stage = approval.target_stage  # "staging" or "production"
            else:
                version.stage = ModelStage.DEVELOPMENT

        await self.session.commit()
        return approval

    def _evaluate_quality_gate(self, metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Verify baseline quality criteria (e.g. F1 >= 0.70, Accuracy >= 0.70, etc.)."""
        if not metrics:
            return {"passed": False, "reason": "No metrics available for evaluation."}

        f1 = metrics.get("f1") or metrics.get("f1_macro") or 0.0
        acc = metrics.get("accuracy") or 0.0
        r2 = metrics.get("r2") or 0.0

        passed = (f1 >= 0.65 or acc >= 0.70 or r2 >= 0.50)
        return {
            "passed": passed,
            "evaluated_metrics": {"f1": f1, "accuracy": acc, "r2": r2},
            "criteria": "F1 >= 0.65 OR Accuracy >= 0.70 OR R2 >= 0.50",
        }
