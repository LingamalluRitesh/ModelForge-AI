-- ModelForge AI - PostgreSQL Migration 0102: Adversarial Debiasing Epoch Tracking DDL

CREATE TABLE IF NOT EXISTS adversarial_debiasing_runs (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version_id VARCHAR(36) NOT NULL REFERENCES model_versions(id) ON DELETE CASCADE,
    protected_attribute VARCHAR(50) NOT NULL,
    lambda_adversary_weight FLOAT DEFAULT 0.5,
    demographic_parity_ratio FLOAT NOT NULL,
    equalized_odds_disparity FLOAT NOT NULL,
    trained_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
