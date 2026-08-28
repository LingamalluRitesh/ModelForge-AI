-- ModelForge AI - PostgreSQL Migration 0080: Immutable SHA-256 Merkle Ledger Signatures DDL

CREATE TABLE IF NOT EXISTS merkle_audit_ledger_roots (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    block_height BIGINT UNIQUE NOT NULL,
    merkle_root_hash VARCHAR(64) NOT NULL,
    previous_block_hash VARCHAR(64) NOT NULL,
    event_count INT NOT NULL,
    sealed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
