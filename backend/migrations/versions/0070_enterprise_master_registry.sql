-- ModelForge AI - PostgreSQL Migration 0070: Enterprise Master MLOps Registry Catalog DDL

CREATE TABLE IF NOT EXISTS enterprise_master_registry_catalog (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id VARCHAR(36) NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    catalog_snapshot_version VARCHAR(50) NOT NULL,
    total_active_models INT NOT NULL,
    total_active_endpoints INT NOT NULL,
    total_monthly_predictions BIGINT NOT NULL,
    compliance_audit_signature TEXT NOT NULL,
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
