"""
ModelForge AI - Database Models Package Export
Exposes all SQLAlchemy models for Alembic discovery and ORM relationship binding.
"""

from app.core.database import Base
from app.models.user import (
    User, Role, Permission, APIKey, LoginHistory, user_roles, role_permissions, RoleEnum
)
from app.models.organization import (
    Organization, OrganizationMember, Team, TeamMember, Invitation
)
from app.models.project import (
    Project, ProjectMember, EnvironmentEnum
)
from app.models.dataset import (
    Dataset, DatasetVersion, DataQualityReport, DataProfileReport
)
from app.models.feature import (
    Feature, FeatureVersion, FeatureView
)
from app.models.experiment import (
    Experiment, ExperimentRun
)
from app.models.training import (
    TrainingJob, AutoMLJob, AutoMLTrial
)
from app.models.model_registry import (
    RegisteredModel, ModelVersion, ModelApprovalRequest, QualityGateConfig, ModelStage
)
from app.models.deployment import (
    Deployment, DeploymentRollbackHistory
)
from app.models.prediction import (
    PredictionLog, BatchInferenceJob
)
from app.models.monitoring import (
    ModelMonitoringMetric, SystemResourceSnapshot
)
from app.models.drift import (
    DriftEvent
)
from app.models.retraining import (
    RetrainingPolicy, RetrainingExecution
)
from app.models.explainability import (
    SHAPExplanation, FairnessAnalysisReport
)
from app.models.pipeline import (
    MLPipeline, PipelineRun, PipelineNodeExecution, WorkflowSchedule
)
from app.models.alert import (
    Alert, NotificationChannel, AuditLog, AlertSeverity, AlertType
)

__all__ = [
    "Base",
    "User", "Role", "Permission", "APIKey", "LoginHistory", "user_roles", "role_permissions", "RoleEnum",
    "Organization", "OrganizationMember", "Team", "TeamMember", "Invitation",
    "Project", "ProjectMember", "EnvironmentEnum",
    "Dataset", "DatasetVersion", "DataQualityReport", "DataProfileReport",
    "Feature", "FeatureVersion", "FeatureView",
    "Experiment", "ExperimentRun",
    "TrainingJob", "AutoMLJob", "AutoMLTrial",
    "RegisteredModel", "ModelVersion", "ModelApprovalRequest", "QualityGateConfig", "ModelStage",
    "Deployment", "DeploymentRollbackHistory",
    "PredictionLog", "BatchInferenceJob",
    "ModelMonitoringMetric", "SystemResourceSnapshot",
    "DriftEvent",
    "RetrainingPolicy", "RetrainingExecution",
    "SHAPExplanation", "FairnessAnalysisReport",
    "MLPipeline", "PipelineRun", "PipelineNodeExecution", "WorkflowSchedule",
    "Alert", "NotificationChannel", "AuditLog", "AlertSeverity", "AlertType",
]
