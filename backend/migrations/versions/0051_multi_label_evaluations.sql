-- ModelForge AI - PostgreSQL Migration 0051: Multi-Label Evaluations DDL

CREATE TABLE IF NOT EXISTS multi_label_model_evaluations (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version_id VARCHAR(36) NOT NULL REFERENCES model_versions(id) ON DELETE CASCADE,
    hamming_loss FLOAT NOT NULL,
    subset_accuracy FLOAT NOT NULL,
    lrap_score FLOAT NOT NULL,
    label_f1_scores JSONB NOT NULL,
    evaluated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
