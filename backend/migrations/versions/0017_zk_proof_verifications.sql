-- ModelForge AI - PostgreSQL Migration 0017: Zero-Knowledge Model Proofs DDL

CREATE TABLE IF NOT EXISTS zk_model_proofs (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version_id VARCHAR(36) NOT NULL REFERENCES model_versions(id) ON DELETE CASCADE,
    proof_root_hash VARCHAR(100) NOT NULL,
    fiat_shamir_challenge VARCHAR(100) NOT NULL,
    verification_status VARCHAR(50) DEFAULT 'VERIFIED',
    circuit_constraints_count INT DEFAULT 124000,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
