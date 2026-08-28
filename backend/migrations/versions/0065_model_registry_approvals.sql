-- ModelForge AI - PostgreSQL Migration 0065: Multi-Signature Model Approvals DDL

CREATE TABLE IF NOT EXISTS model_registry_multisig_approvals (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version_id VARCHAR(36) NOT NULL REFERENCES model_versions(id) ON DELETE CASCADE,
    approver_user_id VARCHAR(36) NOT NULL REFERENCES users(id),
    approval_role VARCHAR(50) NOT NULL, -- DATA_PROTECTION_OFFICER, LEAD_DATA_SCIENTIST, VP_ENGINEERING
    approval_decision VARCHAR(50) DEFAULT 'APPROVED',
    cryptographic_signature TEXT NOT NULL,
    signed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
