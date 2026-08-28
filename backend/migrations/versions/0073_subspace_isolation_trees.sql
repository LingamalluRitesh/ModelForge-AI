-- ModelForge AI - PostgreSQL Migration 0073: Subspace Isolation Forest Forests DDL

CREATE TABLE IF NOT EXISTS subspace_isolation_forests (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version_id VARCHAR(36) NOT NULL REFERENCES model_versions(id) ON DELETE CASCADE,
    num_trees INT DEFAULT 100,
    subsample_size INT DEFAULT 256,
    storage_forest_uri VARCHAR(500) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
