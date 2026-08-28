-- ModelForge AI - PostgreSQL Migration 0011: Model Cards & Regulatory Compliance DDL

-- Model Cards Catalog
CREATE TABLE IF NOT EXISTS model_cards (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version_id VARCHAR(36) UNIQUE NOT NULL REFERENCES model_versions(id) ON DELETE CASCADE,
    compliance_standards JSONB DEFAULT '["EU AI Act", "NIST AI RMF 1.0"]'::jsonb,
    intended_use JSONB NOT NULL,
    quantitative_metrics JSONB NOT NULL,
    fairness_assessment JSONB NOT NULL,
    dataset_provenance JSONB NOT NULL,
    ethical_considerations TEXT,
    human_oversight_protocol TEXT,
    is_signed_off BOOLEAN DEFAULT FALSE,
    signed_off_by_id VARCHAR(36) REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Regulatory Audit Reports
CREATE TABLE IF NOT EXISTS regulatory_audit_reports (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id VARCHAR(36) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    report_type VARCHAR(50) DEFAULT 'eu_ai_act_annex_iv',
    compliance_score FLOAT NOT NULL,
    findings JSONB NOT NULL,
    remediation_steps JSONB,
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
