-- ModelForge AI - PostgreSQL Migration 0084: Enterprise Regulatory Audit Ledger DDL

CREATE TABLE IF NOT EXISTS enterprise_compliance_audit_ledger (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    audit_uuid VARCHAR(36) UNIQUE NOT NULL,
    regulatory_act VARCHAR(100) DEFAULT 'EU_AI_ACT_HIGH_RISK_ANNEX_IV',
    model_version_id VARCHAR(36) NOT NULL REFERENCES model_versions(id),
    compliance_status VARCHAR(50) DEFAULT 'PASSED_CERTIFIED',
    immutable_signature TEXT NOT NULL,
    certified_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
