-- ModelForge AI - PostgreSQL Migration 0064: Enterprise Data Sovereignty & Geo-Fencing DDL

CREATE TABLE IF NOT EXISTS data_sovereignty_regions (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id VARCHAR(36) NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    allowed_storage_regions VARCHAR(50)[] NOT NULL, -- ["us-east-1", "eu-west-1"]
    cross_border_transfer_allowed BOOLEAN DEFAULT FALSE,
    encryption_key_kms_arn VARCHAR(500) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
