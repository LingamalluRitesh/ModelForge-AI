-- ModelForge AI - PostgreSQL Migration 0009: OpenLineage Provenance & Data Flow DDL

-- Lineage Graph Nodes
CREATE TABLE IF NOT EXISTS lineage_nodes (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id VARCHAR(36) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    node_type VARCHAR(50) NOT NULL, -- DATASET, FEATURE_VIEW, EXPERIMENT_RUN, MODEL_VERSION, ENDPOINT
    external_id VARCHAR(36) NOT NULL,
    name VARCHAR(150) NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT uq_lineage_node UNIQUE (project_id, node_type, external_id)
);

-- Lineage Graph Edges
CREATE TABLE IF NOT EXISTS lineage_edges (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id VARCHAR(36) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    source_node_id VARCHAR(36) NOT NULL REFERENCES lineage_nodes(id) ON DELETE CASCADE,
    target_node_id VARCHAR(36) NOT NULL REFERENCES lineage_nodes(id) ON DELETE CASCADE,
    relationship_type VARCHAR(50) NOT NULL, -- TRANSFORMED_INTO, TRAINED_FROM, DEPLOYED_AS, EVALUATED_ON
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT uq_lineage_edge UNIQUE (source_node_id, target_node_id, relationship_type)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_lineage_edges_source ON lineage_edges(source_node_id);
CREATE INDEX IF NOT EXISTS idx_lineage_edges_target ON lineage_edges(target_node_id);
