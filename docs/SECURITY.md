# ModelForge AI — Enterprise Security, RBAC & Compliance Guide

## 1. Authentication Architecture
- **JWT Bearer Authentication**: Signed using HMAC-SHA256 with 24-hour access token expiry and rolling 30-day refresh tokens.
- **API Key Cryptography**: Programmatic keys use `mf_live_` prefixes with constant-time SHA-256 hash comparison in PostgreSQL.
- **Password Security**: Hashed with bcrypt (cost factor 12) with salt generation.
- **MFA / 2FA**: Time-based One-Time Password (TOTP) support compatible with Google Authenticator and 1Password.

---

## 2. Granular Role-Based Access Control (RBAC)

| Role | Organization & Projects | Datasets & Features | Training & AutoML | Model Registry & Approvals | Deployments & Canary | Audit Logs |
|---|---|---|---|---|---|---|
| **SUPER_ADMIN** | Full (Manage All) | Full | Full | Full | Full | Read & Export |
| **ORG_ADMIN** | Manage Org & Projects | Full | Full | Full | Full | Read |
| **ML_ENGINEER** | Project Member | Ingest & Create | Execute | Register & View | Deploy & Canary | Read |
| **DATA_SCIENTIST** | Project Member | Ingest & Create | Execute | Register & View | Read-Only | None |
| **DATA_ENGINEER** | Project Member | Ingest & Transform | None | Read-Only | None | None |
| **MODEL_REVIEWER**| Project Member | Read-Only | Read-Only | Approve & Reject | Read-Only | Read |
| **VIEWER** | Read-Only | Read-Only | Read-Only | Read-Only | Read-Only | None |

---

## 3. Immutable Audit Trails
Every mutation across models, deployments, approvals, and security credentials generates an immutable audit record containing:
- Timestamp (UTC ISO-8601)
- User ID & Principal Email
- Client IP Address & User Agent
- Action Event Code (`deployment.rollback`, `model.approve`)
- Target Resource Type & ID
- Request Correlation ID
