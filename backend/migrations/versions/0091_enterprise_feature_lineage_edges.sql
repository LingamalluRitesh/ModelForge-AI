-- ModelForge AI - PostgreSQL Migration 0091: Feature Lineage Directed Graph Edges DDL

CREATE TABLE IF NOT EXISTS enterprise_feature_lineage_graph (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id VARCHAR(36) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    source_entity VARCHAR(150) NOT NULL,
    destination_feature VARCHAR(150) NOT NULL,
    transformation_dag_id VARCHAR(36) NOT NULL,
    data_freshness_seconds INT DEFAULT 300,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
