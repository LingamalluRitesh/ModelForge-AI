-- ModelForge AI - PostgreSQL Migration 0090: Immutable Enterprise Audit Trail Block Seal DDL

CREATE TABLE IF NOT EXISTS immutable_audit_trail_block_seals (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    block_index BIGINT UNIQUE NOT NULL,
    previous_block_hash VARCHAR(64) NOT NULL,
    merkle_root_hash VARCHAR(64) NOT NULL,
    block_signature TEXT NOT NULL,
    verified_by_kms_key VARCHAR(250) NOT NULL,
    sealed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
