-- ModelForge AI - PostgreSQL Migration 0079: Enterprise Cryptographic License Signatures DDL

CREATE TABLE IF NOT EXISTS enterprise_license_signatures (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id VARCHAR(36) NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    license_tier VARCHAR(50) DEFAULT 'ENTERPRISE_UNLIMITED',
    max_gpu_nodes INT DEFAULT 1024,
    valid_until TIMESTAMP WITH TIME ZONE DEFAULT (NOW() + INTERVAL '365 days'),
    cryptographic_signature TEXT NOT NULL,
    issued_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
