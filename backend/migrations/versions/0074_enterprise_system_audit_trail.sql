-- ModelForge AI - PostgreSQL Migration 0074: Immutable Enterprise Audit Trail DDL

CREATE TABLE IF NOT EXISTS enterprise_system_audit_trail (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    event_uuid VARCHAR(36) UNIQUE NOT NULL,
    principal_actor VARCHAR(150) NOT NULL,
    action_verb VARCHAR(50) NOT NULL,
    resource_urn VARCHAR(250) NOT NULL,
    ip_address VARCHAR(45) NOT NULL,
    cryptographic_sha256_hash VARCHAR(64) NOT NULL,
    logged_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
