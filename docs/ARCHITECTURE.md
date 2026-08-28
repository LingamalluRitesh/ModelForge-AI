# ModelForge AI — System Architecture & Technical Design

## 1. Architectural Philosophy & Principles
ModelForge AI is designed according to enterprise Domain-Driven Design (DDD), Clean Architecture, and microservices-oriented modularity.

The core tenets include:
- **Separation of Concerns**: Strict boundary separation between Presentation (Next.js 14), API Gateway (FastAPI), Domain Services, ML Engine, Data Layer, and Asynchronous Worker Queues.
- **Fail-Safe Resilience**: High-concurrency inference protected by Circuit Breakers, sliding-window rate limiters, health checks, and automated rollbacks.
- **End-to-End Traceability**: Every production inference log is deterministically traced back to a specific Model Version, Experiment Run, Dataset Version, and Git Commit.
- **Stateless API & Distributed Workers**: Core API nodes are completely stateless, delegating long-running computational workloads (AutoML, training, batch scoring, PSI drift calculations) to Celery workers backed by Redis.

---

## 2. Layered Component Decomposition

### 2.1 Presentation & Client Layer
- **Web Application**: Next.js 14 App Router, React 18, TypeScript, Tailwind CSS, Lucide Icons, Recharts, and Zustand state stores.
- **Visual DAG Pipeline Editor**: Interactive vector canvas with drag-and-drop node configuration and topological DAG execution.
- **Programmatic Clients**: ModelForge Python SDK, CLI tools, and REST API consumers.

### 2.2 API Gateway & Security
- **Reverse Proxy & Ingress**: NGINX / Kubernetes Ingress routing with TLS termination and security headers.
- **Authentication & Multi-Tenant RBAC**: Dual-mode authentication (JWT Bearer Token with refresh rotation & SHA-256 hashed API Keys).
- **Immutable Audit Logger**: Intercepts and cryptographically logs every resource mutation.

### 2.3 Core Domain Services Layer
- `AuthService`: Identity, JWT lifecycle, TOTP MFA, and API key generation.
- `OrganizationService` & `ProjectService`: Multi-tenant workspace isolation across Development, Staging, and Production environments.
- `DatasetService`: Ingestion, schema inference, automated data quality scoring, and statistical profiling.
- `FeatureService`: Offline feature store repository and low-latency online hydration views.
- `ExperimentService` & `TrainingService`: Training run execution, hyperparameter logging, and side-by-side run comparisons.
- `AutoMLService`: Automated problem detection, Optuna Bayesian optimization, and trial leaderboards.
- `ModelRegistryService`: Model lifecycle state transitions, quality gate enforcement, and governance approvals.
- `DeploymentService` & `InferenceService`: Microsecond REST serving, Canary rollouts (5%–100%), A/B traffic splitting, and automated rollbacks.
- `MonitoringService` & `DriftService`: Real-time telemetry, $p_{50}/p_{95}/p_{99}$ latency metrics, PSI, KS-test, and Wasserstein drift calculations.
- `RetrainingService`: Autonomous trigger management and candidate challenger vs champion comparisons.
- `PipelineService`: Visual DAG creation, topological sorting, and Celery workflow dispatch.
- `AlertService`: Multi-channel incident dispatching (In-app, Webhooks, Slack, PagerDuty).

### 2.4 ML Algorithms & Compute Engine (`backend/ml_engine/`)
- Pure mathematical and algorithmic implementations utilizing Scikit-learn, XGBoost, LightGBM, PyTorch Deep Learning, Optuna, and SHAP.

### 2.5 Data & Infrastructure Persistence
- **PostgreSQL 16**: Primary relational database with ACID guarantees, normalized foreign keys, and Alembic migrations.
- **Redis 7.2**: In-memory caching, distributed locks, rate-limiting sliding windows, and Celery broker.
- **S3 / MinIO Object Storage**: Immutable model artifacts, pickled pipelines, and raw dataset storage.
- **Prometheus & Grafana**: Time-series telemetry scraping and operational alerting.
