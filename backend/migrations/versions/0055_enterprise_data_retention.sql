-- ModelForge AI - PostgreSQL Migration 0055: Enterprise Data Retention & Purge Policies DDL

CREATE TABLE IF NOT EXISTS enterprise_data_retention_policies (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id VARCHAR(36) NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    table_name VARCHAR(100) NOT NULL,
    retention_days INT DEFAULT 365,
    is_auto_purge_enabled BOOLEAN DEFAULT TRUE,
    last_purged_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
