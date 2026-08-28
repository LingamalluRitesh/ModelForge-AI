-- ModelForge AI - PostgreSQL Migration 0026: Model Fairness Certifications DDL

CREATE TABLE IF NOT EXISTS model_fairness_certifications (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version_id VARCHAR(36) NOT NULL REFERENCES model_versions(id) ON DELETE CASCADE,
    four_fifths_rule_passed BOOLEAN NOT NULL,
    equalized_odds_max_disparity FLOAT NOT NULL,
    disparate_impact_ratios JSONB NOT NULL,
    certified_by_id VARCHAR(36) REFERENCES users(id),
    certification_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
