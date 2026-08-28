# ModelForge AI — REST API Reference Guide

All API endpoints are prefixed with `/api/v1`. Interactive OpenAPI documentation is accessible at `http://localhost:8000/api/v1/docs` or `/redoc`.

---

## 1. Authentication & Security Endpoints

| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `POST` | `/api/v1/auth/register` | Register new user & default enterprise org | No |
| `POST` | `/api/v1/auth/login` | Authenticate with email/password and obtain JWT tokens | No |
| `POST` | `/api/v1/auth/refresh` | Obtain fresh access token using refresh token | No |
| `GET` | `/api/v1/auth/me` | Get profile and active roles of current user | Yes (Bearer / API Key) |
| `POST` | `/api/v1/auth/api-keys` | Generate new hashed API Key for SDK/CLI | Yes |
| `GET` | `/api/v1/auth/api-keys` | List active user API keys | Yes |

---

## 2. Projects & Organizations

| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `GET` | `/api/v1/projects` | List projects in organization | Yes |
| `POST` | `/api/v1/projects` | Create new ML project | Yes |
| `GET` | `/api/v1/projects/{id}` | Get project details by ID | Yes |
| `GET` | `/api/v1/organizations` | List user organizations | Yes |
| `POST` | `/api/v1/organizations` | Create enterprise organization | Yes |

---

## 3. Data Ingestion & Quality

| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `GET` | `/api/v1/datasets?project_id={id}` | List datasets in project | Yes |
| `POST` | `/api/v1/datasets/upload` | Ingest CSV, Parquet, JSON with automated profiling | Yes |
| `GET` | `/api/v1/datasets/versions/{id}/quality` | Get automated Data Quality report | Yes |
| `GET` | `/api/v1/datasets/versions/{id}/profile` | Get statistical distributions & correlation matrix | Yes |

---

## 4. Experiments, Training & AutoML

| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `GET` | `/api/v1/experiments?project_id={id}` | List experiments | Yes |
| `POST` | `/api/v1/experiments` | Create experiment workspace | Yes |
| `GET` | `/api/v1/experiments/{id}/runs` | List training runs | Yes |
| `POST` | `/api/v1/experiments/compare` | Multi-run side-by-side comparison | Yes |
| `POST` | `/api/v1/training/jobs` | Launch model training run | Yes |
| `POST` | `/api/v1/automl/jobs` | Launch AutoML exploration with Optuna HPO | Yes |

---

## 5. Model Registry, Governance & Deployments

| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `GET` | `/api/v1/registry/models?project_id={id}` | List registered models | Yes |
| `POST` | `/api/v1/registry/models` | Create registered model | Yes |
| `POST` | `/api/v1/registry/models/{id}/versions` | Register model version from run | Yes |
| `POST` | `/api/v1/registry/versions/{id}/request-approval` | Submit model version for promotion approval | Yes |
| `POST` | `/api/v1/registry/approvals/{id}/review` | Reviewer approves or rejects promotion | Yes |
| `GET` | `/api/v1/deployments?project_id={id}` | List deployments | Yes |
| `POST` | `/api/v1/deployments` | Create model deployment | Yes |
| `POST` | `/api/v1/deployments/{id}/canary` | Update Canary traffic rollout percentage | Yes |
| `POST` | `/api/v1/deployments/{id}/rollback` | Roll back to previous healthy version | Yes |
| `POST` | `/api/v1/predictions/{endpoint}` | Execute high-throughput real-time prediction | Yes (API Key / Open) |

---

## 6. Observability, Drift, Retraining & Pipelines

| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `GET` | `/api/v1/monitoring/{id}/metrics` | Query timeseries throughput & latency percentiles | Yes |
| `POST` | `/api/v1/drift/analyze` | Execute PSI, KS-test, Wasserstein drift checks | Yes |
| `GET` | `/api/v1/retraining/policies?project_id={id}` | List automated retraining policies | Yes |
| `POST` | `/api/v1/retraining/policies/{id}/trigger` | Trigger retraining pipeline | Yes |
| `POST` | `/api/v1/explainability/shap/global` | Compute global SHAP feature attributions | Yes |
| `POST` | `/api/v1/explainability/fairness` | Audit model algorithmic fairness | Yes |
| `GET` | `/api/v1/pipelines?project_id={id}` | List visual DAG pipelines | Yes |
| `POST` | `/api/v1/pipelines` | Create visual DAG pipeline | Yes |
| `POST` | `/api/v1/pipelines/{id}/run` | Execute visual DAG pipeline | Yes |
| `GET` | `/api/v1/alerts?project_id={id}` | List active alerts and incidents | Yes |
| `GET` | `/api/v1/audit` | Query immutable audit logs | Yes |
