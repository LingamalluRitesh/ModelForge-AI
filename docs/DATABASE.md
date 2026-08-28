# ModelForge AI — Relational Database Schema & Data Models

ModelForge AI uses **PostgreSQL 16** with SQLAlchemy 2.0 async ORM models.

---

## Entity Relationship Overview

```
users (1) ───────────< user_roles >─────────── (N) roles (1) ───────────< role_permissions >─────────── (N) permissions
  │
  ├───< api_keys
  ├───< login_history
  ├───< organization_members (N) ─────────── (1) organizations (1) ─────────── (N) teams
  │                                                    │
  │                                                    └─── (1) projects
  │                                                                │
  ├───< project_members                                            ├───< datasets ───< dataset_versions ───< data_quality_reports
  │                                                                │                                    └───< data_profile_reports
  │                                                                ├───< features ───< feature_versions
  │                                                                │
  │                                                                ├───< experiments ───< experiment_runs ───< registered_models
  │                                                                │                                                │
  │                                                                │                                                └───< model_versions ───< deployments
  │                                                                │                                                                             │
  │                                                                │                                                                             ├───< prediction_logs
  │                                                                │                                                                             ├───< monitoring_metrics
  │                                                                │                                                                             ├───< drift_events
  │                                                                │                                                                             └───< rollback_history
  │                                                                ├───< retraining_policies ───< retraining_executions
  │                                                                ├───< ml_pipelines ───< pipeline_runs ───< node_executions
  │                                                                └───< alerts
  │
  └───< audit_logs
```

---

## Core Tables & Primary Keys

| Table Name | Primary Key | Key Foreign Keys | Purpose |
|---|---|---|---|
| `users` | `id (UUID)` | — | User accounts, hashed passwords, MFA status |
| `roles` | `id (UUID)` | — | RBAC System Roles (Super Admin, ML Engineer, etc.) |
| `permissions` | `id (UUID)` | — | Granular resource-action codes (`models:deploy`) |
| `organizations` | `id (UUID)` | — | Multi-tenant organization tenant workspaces |
| `projects` | `id (UUID)` | `organization_id` | Isolated project environments |
| `datasets` | `id (UUID)` | `project_id`, `created_by_id` | Ingested dataset master catalog |
| `dataset_versions` | `id (UUID)` | `dataset_id` | Immutable dataset versions & storage URIs |
| `data_quality_reports`| `id (UUID)` | `dataset_version_id` | Validation rules & composite quality scores |
| `features` | `id (UUID)` | `project_id`, `source_dataset_id` | Offline/online feature entity store |
| `experiments` | `id (UUID)` | `project_id` | Experiment run groups |
| `experiment_runs` | `id (UUID)` | `experiment_id`, `dataset_version_id` | Parameters, metrics, model artifacts |
| `registered_models` | `id (UUID)` | `project_id` | Versioned model catalog |
| `model_versions` | `id (UUID)` | `registered_model_id`, `experiment_run_id` | Production lifecycle model versions |
| `deployments` | `id (UUID)` | `project_id`, `model_version_id` | Serving endpoints & Canary traffic allocations |
| `prediction_logs` | `id (UUID)` | `deployment_id`, `model_version_id` | Inferences, features, confidences, latencies |
| `drift_events` | `id (UUID)` | `deployment_id` | Statistical PSI, KS, and Wasserstein drift logs |
| `retraining_policies` | `id (UUID)` | `project_id`, `target_registered_model_id` | Autonomous retraining trigger rules |
| `ml_pipelines` | `id (UUID)` | `project_id` | Visual DAG definitions & topological workflows |
| `alerts` | `id (UUID)` | `project_id` | Real-time incident logs & acknowledgements |
| `audit_logs` | `id (UUID)` | `user_id`, `project_id` | Cryptographically immutable audit records |
