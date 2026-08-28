-- ModelForge AI - PostgreSQL Migration 0100: Enterprise Master Compliance Certificate DDL

CREATE TABLE IF NOT EXISTS enterprise_master_compliance_certificates (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id VARCHAR(36) NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    certificate_serial_number VARCHAR(100) UNIQUE NOT NULL,
    compliance_frameworks VARCHAR(100)[] NOT NULL, -- EU_AI_ACT, NIST_AI_RMF, ISO_42001, SOC2_TYPE_II
    overall_trust_score FLOAT NOT NULL,
    certified_by VARCHAR(150) DEFAULT 'ModelForge Autonomous Governance Engine v2.4',
    cryptographic_sha256_merkle_root VARCHAR(64) NOT NULL,
    issued_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
