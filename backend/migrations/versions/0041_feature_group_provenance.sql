-- ModelForge AI - PostgreSQL Migration 0041: Feature Group Provenance Graph DDL

CREATE TABLE IF NOT EXISTS feature_provenance_edges (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id VARCHAR(36) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    source_table VARCHAR(150) NOT NULL,
    target_feature_name VARCHAR(150) NOT NULL,
    transformation_sql TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
