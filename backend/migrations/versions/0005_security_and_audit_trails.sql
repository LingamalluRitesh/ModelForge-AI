-- ModelForge AI - PostgreSQL Migration 0005: Security, Audit Trails & Cost Tracking DDL

-- Security Audit Logs
CREATE TABLE IF NOT EXISTS security_audit_logs (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id VARCHAR(36) NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    user_id VARCHAR(36) REFERENCES users(id),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100) NOT NULL,
    resource_id VARCHAR(100) NOT NULL,
    ip_address VARCHAR(45),
    user_agent TEXT,
    payload JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Cloud Cost Allocations
CREATE TABLE IF NOT EXISTS cloud_cost_records (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id VARCHAR(36) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    deployment_id VARCHAR(36) REFERENCES deployments(id),
    billing_period VARCHAR(20) NOT NULL, -- e.g. 2026-08
    compute_cost_usd FLOAT DEFAULT 0.0,
    storage_cost_usd FLOAT DEFAULT 0.0,
    network_cost_usd FLOAT DEFAULT 0.0,
    quantization_savings_usd FLOAT DEFAULT 0.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_audit_org_created ON security_audit_logs(organization_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_cost_project ON cloud_cost_records(project_id, billing_period);
