-- ModelForge AI - PostgreSQL Migration 0095: Final Enterprise Governance Certifications DDL

CREATE TABLE IF NOT EXISTS final_enterprise_governance_certifications (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id VARCHAR(36) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    model_version_id VARCHAR(36) NOT NULL REFERENCES model_versions(id),
    nist_ai_rmf_passed BOOLEAN DEFAULT TRUE,
    eu_ai_act_annex_iv_passed BOOLEAN DEFAULT TRUE,
    iso_42001_certified BOOLEAN DEFAULT TRUE,
    lead_architect_signoff VARCHAR(150) NOT NULL,
    dpo_signoff VARCHAR(150) NOT NULL,
    immutable_root_hash VARCHAR(64) NOT NULL,
    certified_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
