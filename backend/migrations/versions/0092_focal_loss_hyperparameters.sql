-- ModelForge AI - PostgreSQL Migration 0092: Focal Loss Hyperparameter Training Manifests DDL

CREATE TABLE IF NOT EXISTS focal_loss_training_manifests (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version_id VARCHAR(36) NOT NULL REFERENCES model_versions(id) ON DELETE CASCADE,
    gamma_modulating_factor FLOAT DEFAULT 2.0,
    alpha_class_weight FLOAT DEFAULT 0.25,
    hard_negative_ratio FLOAT NOT NULL,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
