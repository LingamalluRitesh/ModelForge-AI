-- ModelForge AI - PostgreSQL Migration 0071: Federated Hardware TPM Attestations DDL

CREATE TABLE IF NOT EXISTS federated_hardware_attestations (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id VARCHAR(100) NOT NULL,
    tpm_quote_signature TEXT NOT NULL,
    pcr_digest_hash VARCHAR(64) NOT NULL,
    hardware_security_level VARCHAR(50) DEFAULT 'HARDWARE_ENCLAVE_SGX',
    is_verified BOOLEAN DEFAULT TRUE,
    attested_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
