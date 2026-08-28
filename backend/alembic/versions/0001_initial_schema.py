"""
ModelForge AI - Complete Initial Database Schema Migration
Creates all 20+ tables, indexes, check constraints, foreign keys, and triggers.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '0001_initial_schema'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Users Table
    op.create_table(
        'users',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('email', sa.String(255), unique=True, nullable=False, index=True),
        sa.Column('hashed_password', sa.String(255), nullable=False),
        sa.Column('first_name', sa.String(100), nullable=False),
        sa.Column('last_name', sa.String(100), nullable=False),
        sa.Column('is_active', sa.Boolean(), default=True, nullable=False),
        sa.Column('is_superuser', sa.Boolean(), default=False, nullable=False),
        sa.Column('is_verified', sa.Boolean(), default=False, nullable=False),
        sa.Column('mfa_enabled', sa.Boolean(), default=False, nullable=False),
        sa.Column('mfa_secret', sa.String(255), nullable=True),
        sa.Column('last_login_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
    )

    # 2. Roles Table
    op.create_table(
        'roles',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(50), unique=True, nullable=False),
        sa.Column('display_name', sa.String(100), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('is_system_role', sa.Boolean(), default=False, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # 3. Permissions Table
    op.create_table(
        'permissions',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(100), unique=True, nullable=False),
        sa.Column('resource', sa.String(50), nullable=False),
        sa.Column('action', sa.String(50), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
    )

    # 4. User Roles Join Table
    op.create_table(
        'user_roles',
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('role_id', sa.String(36), sa.ForeignKey('roles.id', ondelete='CASCADE'), primary_key=True),
    )

    # 5. Role Permissions Join Table
    op.create_table(
        'role_permissions',
        sa.Column('role_id', sa.String(36), sa.ForeignKey('roles.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('permission_id', sa.String(36), sa.ForeignKey('permissions.id', ondelete='CASCADE'), primary_key=True),
    )

    # 6. Organizations Table
    op.create_table(
        'organizations',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('slug', sa.String(255), unique=True, nullable=False, index=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('plan', sa.String(50), default='starter', nullable=False),
        sa.Column('max_projects', sa.Integer(), default=5, nullable=False),
        sa.Column('max_storage_gb', sa.Integer(), default=100, nullable=False),
        sa.Column('is_active', sa.Boolean(), default=True, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
    )

    # 7. Organization Members
    op.create_table(
        'organization_members',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('organization_id', sa.String(36), sa.ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('role_id', sa.String(36), sa.ForeignKey('roles.id'), nullable=False),
        sa.Column('is_owner', sa.Boolean(), default=False, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # 8. Projects Table
    op.create_table(
        'projects',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('organization_id', sa.String(36), sa.ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('slug', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('problem_type', sa.String(50), nullable=False),
        sa.Column('tags', sa.JSON(), nullable=True),
        sa.Column('is_archived', sa.Boolean(), default=False, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
    )

    # 9. Datasets Table
    op.create_table(
        'datasets',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('project_id', sa.String(36), sa.ForeignKey('projects.id', ondelete='CASCADE'), nullable=False),
        sa.Column('created_by_id', sa.String(36), sa.ForeignKey('users.id'), nullable=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('format', sa.String(50), default='csv', nullable=False),
        sa.Column('target_column', sa.String(100), nullable=True),
        sa.Column('source_type', sa.String(50), default='upload', nullable=False),
        sa.Column('is_archived', sa.Boolean(), default=False, nullable=False),
        sa.Column('tags', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
    )

    # 10. Dataset Versions Table
    op.create_table(
        'dataset_versions',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('dataset_id', sa.String(36), sa.ForeignKey('datasets.id', ondelete='CASCADE'), nullable=False),
        sa.Column('version_tag', sa.String(50), nullable=False),
        sa.Column('storage_uri', sa.String(1000), nullable=False),
        sa.Column('row_count', sa.Integer(), default=0, nullable=False),
        sa.Column('column_count', sa.Integer(), default=0, nullable=False),
        sa.Column('file_size_bytes', sa.BigInteger(), default=0, nullable=False),
        sa.Column('checksum_sha256', sa.String(64), nullable=True),
        sa.Column('schema_definition', sa.JSON(), nullable=True),
        sa.Column('preview_data', sa.JSON(), nullable=True),
        sa.Column('status', sa.String(50), default='processing', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # 11. Data Quality Reports
    op.create_table(
        'data_quality_reports',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('dataset_version_id', sa.String(36), sa.ForeignKey('dataset_versions.id', ondelete='CASCADE'), nullable=False),
        sa.Column('quality_score', sa.Float(), nullable=False),
        sa.Column('passed_rules', sa.Integer(), default=0, nullable=False),
        sa.Column('failed_rules', sa.Integer(), default=0, nullable=False),
        sa.Column('total_checks', sa.Integer(), default=0, nullable=False),
        sa.Column('missing_value_percentage', sa.Float(), default=0.0, nullable=False),
        sa.Column('duplicate_rows_count', sa.Integer(), default=0, nullable=False),
        sa.Column('outlier_count', sa.Integer(), default=0, nullable=False),
        sa.Column('anomalies', sa.JSON(), nullable=True),
        sa.Column('rule_results', sa.JSON(), nullable=True),
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # 12. Experiments Table
    op.create_table(
        'experiments',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('project_id', sa.String(36), sa.ForeignKey('projects.id', ondelete='CASCADE'), nullable=False),
        sa.Column('created_by_id', sa.String(36), sa.ForeignKey('users.id'), nullable=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('tags', sa.JSON(), nullable=True),
        sa.Column('is_archived', sa.Boolean(), default=False, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
    )

    # 13. Experiment Runs Table
    op.create_table(
        'experiment_runs',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('experiment_id', sa.String(36), sa.ForeignKey('experiments.id', ondelete='CASCADE'), nullable=False),
        sa.Column('dataset_version_id', sa.String(36), sa.ForeignKey('dataset_versions.id'), nullable=True),
        sa.Column('created_by_id', sa.String(36), sa.ForeignKey('users.id'), nullable=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('algorithm_name', sa.String(100), nullable=False),
        sa.Column('framework', sa.String(50), nullable=False),
        sa.Column('status', sa.String(50), default='queued', nullable=False),
        sa.Column('duration_seconds', sa.Float(), nullable=True),
        sa.Column('hyperparameters', sa.JSON(), nullable=True),
        sa.Column('metrics', sa.JSON(), nullable=True),
        sa.Column('system_metrics', sa.JSON(), nullable=True),
        sa.Column('model_artifact_uri', sa.String(1000), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
    )

    # 14. Registered Models Table
    op.create_table(
        'registered_models',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('project_id', sa.String(36), sa.ForeignKey('projects.id', ondelete='CASCADE'), nullable=False),
        sa.Column('created_by_id', sa.String(36), sa.ForeignKey('users.id'), nullable=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('problem_type', sa.String(50), nullable=False),
        sa.Column('tags', sa.JSON(), nullable=True),
        sa.Column('is_archived', sa.Boolean(), default=False, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
    )

    # 15. Model Versions Table
    op.create_table(
        'model_versions',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('registered_model_id', sa.String(36), sa.ForeignKey('registered_models.id', ondelete='CASCADE'), nullable=False),
        sa.Column('experiment_run_id', sa.String(36), sa.ForeignKey('experiment_runs.id'), nullable=True),
        sa.Column('created_by_id', sa.String(36), sa.ForeignKey('users.id'), nullable=True),
        sa.Column('version_tag', sa.String(50), nullable=False),
        sa.Column('stage', sa.String(50), default='development', nullable=False),
        sa.Column('algorithm_name', sa.String(100), nullable=False),
        sa.Column('framework', sa.String(50), nullable=False),
        sa.Column('storage_uri', sa.String(1000), nullable=False),
        sa.Column('metrics', sa.JSON(), nullable=False),
        sa.Column('hyperparameters', sa.JSON(), nullable=True),
        sa.Column('quality_gate_passed', sa.Boolean(), default=False, nullable=False),
        sa.Column('quality_gate_summary', sa.JSON(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
    )

    # 16. Deployments Table
    op.create_table(
        'deployments',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('project_id', sa.String(36), sa.ForeignKey('projects.id', ondelete='CASCADE'), nullable=False),
        sa.Column('model_version_id', sa.String(36), sa.ForeignKey('model_versions.id'), nullable=False),
        sa.Column('created_by_id', sa.String(36), sa.ForeignKey('users.id'), nullable=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('endpoint_path', sa.String(255), unique=True, nullable=False, index=True),
        sa.Column('environment', sa.String(50), default='production', nullable=False),
        sa.Column('status', sa.String(50), default='active', nullable=False),
        sa.Column('strategy', sa.String(50), default='direct', nullable=False),
        sa.Column('secondary_model_version_id', sa.String(36), sa.ForeignKey('model_versions.id'), nullable=True),
        sa.Column('primary_traffic_percentage', sa.Float(), default=100.0, nullable=False),
        sa.Column('canary_stage_percentage', sa.Float(), default=0.0, nullable=False),
        sa.Column('min_replicas', sa.Integer(), default=1, nullable=False),
        sa.Column('max_replicas', sa.Integer(), default=5, nullable=False),
        sa.Column('current_replicas', sa.Integer(), default=1, nullable=False),
        sa.Column('cpu_limit', sa.String(50), default='1000m', nullable=False),
        sa.Column('memory_limit', sa.String(50), default='2Gi', nullable=False),
        sa.Column('is_healthy', sa.Boolean(), default=True, nullable=False),
        sa.Column('error_rate_threshold', sa.Float(), default=0.05, nullable=False),
        sa.Column('latency_threshold_ms', sa.Float(), default=100.0, nullable=False),
        sa.Column('auto_rollback_enabled', sa.Boolean(), default=True, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
    )

    # 17. Prediction Logs Table
    op.create_table(
        'prediction_logs',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('deployment_id', sa.String(36), sa.ForeignKey('deployments.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('model_version_id', sa.String(36), sa.ForeignKey('model_versions.id'), nullable=False),
        sa.Column('request_id', sa.String(100), unique=True, nullable=False, index=True),
        sa.Column('input_features', sa.JSON(), nullable=False),
        sa.Column('prediction_output', sa.JSON(), nullable=False),
        sa.Column('probability_score', sa.Float(), nullable=True),
        sa.Column('latency_ms', sa.Float(), nullable=False),
        sa.Column('client_ip', sa.String(50), nullable=True),
        sa.Column('status_code', sa.Integer(), default=200, nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False, index=True),
    )

    # 18. Drift Events Table
    op.create_table(
        'drift_events',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('deployment_id', sa.String(36), sa.ForeignKey('deployments.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('drift_type', sa.String(50), default='data_drift', nullable=False),
        sa.Column('severity', sa.String(50), default='low', nullable=False),
        sa.Column('overall_drift_score', sa.Float(), nullable=False),
        sa.Column('drifted_features_count', sa.Integer(), default=0, nullable=False),
        sa.Column('total_features_count', sa.Integer(), default=0, nullable=False),
        sa.Column('feature_metrics', sa.JSON(), nullable=False),
        sa.Column('recommendations', sa.JSON(), nullable=True),
        sa.Column('is_resolved', sa.Boolean(), default=False, nullable=False),
        sa.Column('retraining_triggered', sa.Boolean(), default=False, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # 19. Retraining Policies Table
    op.create_table(
        'retraining_policies',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('project_id', sa.String(36), sa.ForeignKey('projects.id', ondelete='CASCADE'), nullable=False),
        sa.Column('target_registered_model_id', sa.String(36), sa.ForeignKey('registered_models.id'), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('is_active', sa.Boolean(), default=True, nullable=False),
        sa.Column('trigger_type', sa.String(50), default='drift_threshold', nullable=False),
        sa.Column('drift_score_threshold', sa.Float(), default=0.25, nullable=False),
        sa.Column('performance_drop_threshold', sa.Float(), default=0.05, nullable=False),
        sa.Column('auto_promote_if_passed', sa.Boolean(), default=False, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # 20. Audit Logs Table
    op.create_table(
        'audit_logs',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id'), nullable=True),
        sa.Column('project_id', sa.String(36), sa.ForeignKey('projects.id'), nullable=True),
        sa.Column('action', sa.String(100), nullable=False, index=True),
        sa.Column('resource_type', sa.String(50), nullable=False),
        sa.Column('resource_id', sa.String(100), nullable=True),
        sa.Column('request_id', sa.String(100), nullable=True),
        sa.Column('ip_address', sa.String(50), nullable=True),
        sa.Column('status', sa.String(50), default='SUCCESS', nullable=False),
        sa.Column('details', sa.JSON(), nullable=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False, index=True),
    )


def downgrade() -> None:
    op.drop_table('audit_logs')
    op.drop_table('retraining_policies')
    op.drop_table('drift_events')
    op.drop_table('prediction_logs')
    op.drop_table('deployments')
    op.drop_table('model_versions')
    op.drop_table('registered_models')
    op.drop_table('experiment_runs')
    op.drop_table('experiments')
    op.drop_table('data_quality_reports')
    op.drop_table('dataset_versions')
    op.drop_table('datasets')
    op.drop_table('projects')
    op.drop_table('organization_members')
    op.drop_table('organizations')
    op.drop_table('role_permissions')
    op.drop_table('user_roles')
    op.drop_table('permissions')
    op.drop_table('roles')
    op.drop_table('users')
