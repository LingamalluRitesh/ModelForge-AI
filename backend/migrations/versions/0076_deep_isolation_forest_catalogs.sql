-- ModelForge AI - PostgreSQL Migration 0076: Deep Isolation Forest Ensembles DDL

CREATE TABLE IF NOT EXISTS deep_isolation_forest_ensembles (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version_id VARCHAR(36) NOT NULL REFERENCES model_versions(id) ON DELETE CASCADE,
    num_neural_projections INT DEFAULT 50,
    projection_dimension INT DEFAULT 32,
    contamination_ratio FLOAT DEFAULT 0.05,
    storage_weights_uri VARCHAR(500) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
