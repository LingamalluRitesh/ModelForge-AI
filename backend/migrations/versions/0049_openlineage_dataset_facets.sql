-- ModelForge AI - PostgreSQL Migration 0049: OpenLineage Dataset Facets DDL

CREATE TABLE IF NOT EXISTS openlineage_dataset_facets (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id VARCHAR(36) NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    facet_key VARCHAR(100) NOT NULL,
    facet_data JSONB NOT NULL,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
