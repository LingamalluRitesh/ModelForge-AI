-- ModelForge AI - PostgreSQL Migration 0030: Continuous Retraining Audits DDL

CREATE TABLE IF NOT EXISTS continuous_retraining_audits (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id VARCHAR(36) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    from_model_version VARCHAR(50) NOT NULL,
    to_model_version VARCHAR(50) NOT NULL,
    trigger_reason VARCHAR(100) NOT NULL,
    baseline_psi FLOAT NOT NULL,
    retrained_f1_score FLOAT NOT NULL,
    canary_promotion_status VARCHAR(50) DEFAULT 'PROMOTED',
    executed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
