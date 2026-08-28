-- ModelForge AI - PostgreSQL Migration 0082: GrowNet Boosting Stage Manifests DDL

CREATE TABLE IF NOT EXISTS grownet_boosting_stages (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version_id VARCHAR(36) NOT NULL REFERENCES model_versions(id) ON DELETE CASCADE,
    stage_index INT NOT NULL,
    weak_learner_type VARCHAR(50) DEFAULT 'shallow_mlp_2layer',
    stage_loss FLOAT NOT NULL,
    weights_binary_s3_uri VARCHAR(500) NOT NULL,
    trained_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
