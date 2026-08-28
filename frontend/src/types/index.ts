/**
 * ModelForge AI - TypeScript Enterprise Type Definitions
 */

export type UserRole = "SUPER_ADMIN" | "ORG_ADMIN" | "ML_ENGINEER" | "DATA_SCIENTIST" | "DATA_ENGINEER" | "MODEL_REVIEWER" | "BUSINESS_ANALYST" | "VIEWER";

export interface User {
  id: string;
  email: string;
  first_name: string;
  last_name: string;
  full_name: string;
  is_active: boolean;
  is_superuser: boolean;
  is_verified: boolean;
  mfa_enabled: boolean;
  roles: { id: string; name: UserRole; display_name: string }[];
  created_at: string;
  last_login_at?: string;
}

export interface Organization {
  id: string;
  name: string;
  slug: string;
  description?: string;
  plan: string;
  max_projects: number;
  max_storage_gb: number;
  is_active: boolean;
  created_at: string;
}

export interface Project {
  id: string;
  organization_id: string;
  name: string;
  slug: string;
  description?: string;
  problem_type: "classification" | "regression" | "clustering" | "deep_learning";
  tags?: string[];
  is_archived: boolean;
  created_at: string;
  updated_at: string;
}

export interface DatasetVersion {
  id: string;
  dataset_id: string;
  version_tag: string;
  storage_uri: string;
  row_count: number;
  column_count: number;
  file_size_bytes: number;
  checksum_sha256?: string;
  schema_definition?: Record<string, string>;
  preview_data?: Record<string, any>[];
  status: "processing" | "ready" | "failed";
  created_at: string;
}

export interface Dataset {
  id: string;
  project_id: string;
  name: string;
  description?: string;
  format: string;
  target_column?: string;
  source_type: string;
  is_archived: boolean;
  tags?: string[];
  created_at: string;
  updated_at: string;
  latest_version?: DatasetVersion;
}

export interface DataQualityReport {
  id: string;
  dataset_version_id: string;
  quality_score: number;
  passed_rules: number;
  failed_rules: number;
  total_checks: number;
  missing_value_percentage: number;
  duplicate_rows_count: number;
  outlier_count: number;
  anomalies?: Record<string, any>;
  rule_results?: { rule: string; passed: boolean; details: string; severity: string }[];
  summary?: string;
  created_at: string;
}

export interface DataProfileReport {
  id: string;
  dataset_version_id: string;
  column_stats: Record<string, any>;
  correlations?: { columns: string[]; matrix: number[][] };
  histograms?: Record<string, any[]>;
  missingness_matrix?: Record<string, any>;
  created_at: string;
}

export interface Feature {
  id: string;
  project_id: string;
  name: string;
  entity_name: string;
  description?: string;
  data_type: string;
  transformation_type: string;
  transformation_params?: Record<string, any>;
  status: "active" | "deprecated" | "archived";
  tags?: string[];
  created_at: string;
  updated_at: string;
}

export interface ExperimentRun {
  id: string;
  experiment_id: string;
  name: string;
  dataset_version_id?: string;
  algorithm_name: string;
  framework: string;
  status: "queued" | "running" | "completed" | "failed";
  duration_seconds?: number;
  hyperparameters?: Record<string, any>;
  metrics?: Record<string, any>;
  system_metrics?: Record<string, any>;
  model_artifact_uri?: string;
  created_at: string;
  completed_at?: string;
}

export interface Experiment {
  id: string;
  project_id: string;
  name: string;
  description?: string;
  tags?: string[];
  is_archived: boolean;
  created_at: string;
  updated_at: string;
  runs_count?: number;
}

export interface ModelVersion {
  id: string;
  registered_model_id: string;
  version_tag: string;
  stage: "development" | "candidate" | "validation" | "approved" | "staging" | "production" | "archived";
  experiment_run_id?: string;
  algorithm_name: string;
  framework: string;
  storage_uri: string;
  metrics: Record<string, any>;
  hyperparameters?: Record<string, any>;
  quality_gate_passed: boolean;
  quality_gate_summary?: Record<string, any>;
  description?: string;
  created_at: string;
  updated_at: string;
}

export interface RegisteredModel {
  id: string;
  project_id: string;
  name: string;
  description?: string;
  problem_type: string;
  tags?: string[];
  is_archived: boolean;
  created_at: string;
  updated_at: string;
  versions: ModelVersion[];
}

export interface Deployment {
  id: string;
  project_id: string;
  name: string;
  endpoint_path: string;
  environment: "development" | "staging" | "production";
  status: "active" | "paused" | "scaling" | "updating" | "terminated";
  model_version_id: string;
  strategy: "direct" | "canary" | "ab_test" | "shadow";
  secondary_model_version_id?: string;
  primary_traffic_percentage: number;
  canary_stage_percentage: number;
  min_replicas: number;
  max_replicas: number;
  current_replicas: number;
  cpu_limit: string;
  memory_limit: string;
  is_healthy: boolean;
  error_rate_threshold: number;
  latency_threshold_ms: number;
  auto_rollback_enabled: boolean;
  created_at: string;
  updated_at: string;
}

export interface ModelMonitoringMetric {
  id: string;
  deployment_id: string;
  timestamp: string;
  request_count: number;
  error_count: number;
  error_rate: number;
  latency_p50: number;
  latency_p95: number;
  latency_p99: number;
  observed_accuracy?: number;
  observed_f1?: number;
}

export interface DriftEvent {
  id: string;
  deployment_id: string;
  drift_type: string;
  severity: "low" | "medium" | "warning" | "high" | "critical";
  overall_drift_score: number;
  drifted_features_count: number;
  total_features_count: number;
  feature_metrics: Record<string, {
    feature_name: string;
    drift_score: number;
    algorithm_used: string;
    p_value?: number;
    has_drifted: boolean;
    severity: string;
  }>;
  recommendations: string[];
  is_resolved: boolean;
  retraining_triggered: boolean;
  created_at: string;
}

export interface RetrainingPolicy {
  id: string;
  project_id: string;
  name: string;
  is_active: boolean;
  trigger_type: string;
  drift_score_threshold: number;
  performance_drop_threshold: number;
  target_registered_model_id: string;
  auto_promote_if_passed: boolean;
  created_at: string;
}

export interface MLPipeline {
  id: string;
  project_id: string;
  name: string;
  description?: string;
  dag_definition: {
    nodes: Array<{
      id: string;
      name: string;
      type: string;
      position: { x: number; y: number };
      config: Record<string, any>;
    }>;
    edges: Array<{
      id: string;
      source: string;
      target: string;
    }>;
  };
  is_active: boolean;
  tags?: string[];
  created_at: string;
  updated_at: string;
}

export interface Alert {
  id: string;
  project_id: string;
  title: string;
  message: string;
  alert_type: string;
  severity: "info" | "warning" | "critical";
  source_resource_type: string;
  source_resource_id: string;
  is_acknowledged: boolean;
  is_resolved: boolean;
  created_at: string;
}

export interface AuditLog {
  id: string;
  user_id?: string;
  project_id?: string;
  action: string;
  resource_type: string;
  resource_id?: string;
  request_id?: string;
  ip_address?: string;
  status: "SUCCESS" | "FAILURE" | "FORBIDDEN";
  details?: Record<string, any>;
  timestamp: string;
}
