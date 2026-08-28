-- ModelForge AI - PostgreSQL Migration 0040: Continuous Retraining Autonomous Policies DDL

CREATE TABLE IF NOT EXISTS autonomous_retraining_policies (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id VARCHAR(36) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    psi_drift_threshold FLOAT DEFAULT 0.20,
    ks_test_pvalue_threshold FLOAT DEFAULT 0.05,
    min_holdout_accuracy FLOAT DEFAULT 0.88,
    cooldown_period_hours INT DEFAULT 24,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
