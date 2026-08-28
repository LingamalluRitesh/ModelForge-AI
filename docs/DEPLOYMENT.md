# ModelForge AI — Kubernetes & Multi-Cloud Production Deployment

## 1. Kubernetes Production Architecture
In production, ModelForge AI runs as highly-scalable, fault-tolerant Kubernetes deployments with Horizontal Pod Autoscaling (HPA).

### Architecture Components:
- **API Pods (`modelforge-api`)**: Stateless FastAPI instances scaled from 3 to 15 pods based on CPU/traffic utilization.
- **Worker Pods (`modelforge-worker`)**: Celery distributed workers handling ML training and batch inference.
- **Frontend Pods (`modelforge-frontend`)**: Next.js SSR and client dashboard instances.
- **Storage**: Managed AWS RDS PostgreSQL / Cloud SQL and AWS S3 / Google Cloud Storage.
- **Cache**: Managed AWS ElastiCache Redis / MemoryStore.

---

## 2. Helm Chart Deployment

```bash
# 1. Add ModelForge Helm repo or use local chart
cd infrastructure/helm

# 2. Deploy to Kubernetes namespace
helm upgrade --install modelforge-platform . \
  --namespace modelforge \
  --create-namespace \
  --values values.yaml
```

---

## 3. Terraform Multi-Cloud Infrastructure Provisioning

```bash
cd infrastructure/terraform

# Initialize Terraform providers
terraform init

# Review execution plan
terraform plan

# Apply infrastructure provisioning (S3, RDS, EKS, Redis)
terraform apply -auto-approve
```
