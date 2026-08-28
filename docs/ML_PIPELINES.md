# ModelForge AI — ML Pipelines & Visual DAG Workflows

## 1. Visual DAG Pipeline Builder Overview
ModelForge AI provides an interactive, visual DAG (Directed Acyclic Graph) workflow engine. Pipelines orchestrate automated data validation, feature scaling, model training, evaluation, quality gate checking, and deployment.

---

## 2. Supported DAG Node Types

| Node Type | Icon | Inputs | Outputs | Description |
|---|---|---|---|---|
| `dataset` | Database | File / URI | DataFrame | Ingests and version-pins a tabular dataset. |
| `validation` | CheckCircle | DataFrame | Quality Report | Runs 15+ automated data quality checks and enforces score thresholds. |
| `feature_eng` | Layers | Clean Data | Scaled Matrix | Applies Imputation, Scalers, Categorical Encoding, and Feature Selection. |
| `train` | Cpu | Matrix, Hyperparams | Model Artifact | Fits XGBoost, LightGBM, Random Forest, or PyTorch neural network. |
| `evaluate` | Eye | Model, Test Data | Metrics / SHAP | Calculates $F_1$, Accuracy, ROC-AUC, Confusion Matrix, and SHAP attributions. |
| `register` | Archive | Model, Metrics | Model Version | Registers model version in Model Registry. |
| `quality_gate`| ShieldCheck | Model Version | Pass / Fail | Evaluates governance rules ($F_1 \ge 0.85$, Quality $\ge 90\%$). |
| `deploy` | Rocket | Model Version | Endpoint URL | Deploys model to Staging or Production Canary ($10\%$). |
| `notify` | Bell | Status Payload | Webhook Dispatch | Dispatches notifications to Slack, PagerDuty, or Email. |

---

## 3. Execution Engine & Topological Sort
When a pipeline is executed:
1. The DAG definition JSON `{ "nodes": [...], "edges": [...] }` is parsed.
2. In-degree dependencies are evaluated using Kahn's algorithm for topological sorting.
3. Node tasks are dispatched asynchronously to Celery worker pools.
4. Intermediate state and node outputs are persisted in PostgreSQL.
5. If any blocking Quality Gate fails, downstream Deployment nodes are safely halted.
