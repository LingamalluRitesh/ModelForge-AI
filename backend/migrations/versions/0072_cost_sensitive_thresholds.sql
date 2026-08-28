-- ModelForge AI - PostgreSQL Migration 0072: Cost-Sensitive Optimization Thresholds DDL

CREATE TABLE IF NOT EXISTS cost_sensitive_threshold_configs (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version_id VARCHAR(36) NOT NULL REFERENCES model_versions(id) ON DELETE CASCADE,
    optimal_threshold FLOAT NOT NULL,
    cost_fp_usd FLOAT NOT NULL,
    cost_fn_usd FLOAT NOT NULL,
    expected_savings_usd FLOAT NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
