-- ModelForge AI - PostgreSQL Migration 0046: Dataset Schema Evolution Catalog DDL

CREATE TABLE IF NOT EXISTS schema_evolution_versions (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id VARCHAR(36) NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    version_number INT NOT NULL,
    schema_diff JSONB NOT NULL,
    is_backward_compatible BOOLEAN DEFAULT TRUE,
    committed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
