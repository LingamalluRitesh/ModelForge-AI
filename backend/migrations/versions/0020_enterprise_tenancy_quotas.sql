-- ModelForge AI - PostgreSQL Migration 0020: Multi-Tenant Quotas & Rate Limits DDL

CREATE TABLE IF NOT EXISTS enterprise_tenant_quotas (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id VARCHAR(36) UNIQUE NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    max_concurrent_inferences INT DEFAULT 5000,
    max_gpus_allocated INT DEFAULT 64,
    monthly_budget_usd FLOAT DEFAULT 15000.0,
    current_spend_usd FLOAT DEFAULT 3300.0,
    rate_limit_qps INT DEFAULT 2500,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
