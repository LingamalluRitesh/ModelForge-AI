-- ModelForge AI - PostgreSQL Migration 0050: Compliance Audit Certifications DDL

CREATE TABLE IF NOT EXISTS compliance_certifications (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id VARCHAR(36) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    standard_name VARCHAR(100) NOT NULL, -- NIST_AI_RMF, EU_AI_ACT, ISO_42001
    audit_score FLOAT NOT NULL,
    signoff_hash VARCHAR(64) NOT NULL,
    certified_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
