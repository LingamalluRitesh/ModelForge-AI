-- ModelForge AI - PostgreSQL Migration 0086: Model Lineage Dependency Graph DDL

CREATE TABLE IF NOT EXISTS model_lineage_dependency_nodes (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id VARCHAR(36) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    node_type VARCHAR(50) NOT NULL, -- DATASET, FEATURE_VIEW, EXPERIMENT, MODEL_VERSION, ENDPOINT
    node_urn VARCHAR(250) UNIQUE NOT NULL,
    metadata_payload JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS model_lineage_dependency_edges (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    source_node_urn VARCHAR(250) NOT NULL,
    target_node_urn VARCHAR(250) NOT NULL,
    edge_type VARCHAR(50) DEFAULT 'TRANSFORMS_INTO',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
