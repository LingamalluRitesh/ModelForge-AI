-- ModelForge AI - PostgreSQL Migration 0063: DARTS Neural Architecture Supernets DDL

CREATE TABLE IF NOT EXISTS darts_nas_supernets (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id VARCHAR(36) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    supernet_weights_uri VARCHAR(500) NOT NULL,
    best_genotype_json JSONB NOT NULL,
    final_val_accuracy FLOAT NOT NULL,
    search_duration_seconds FLOAT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
