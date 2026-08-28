-- ModelForge AI - PostgreSQL Migration 0068: Metapath2Vec Schema Catalogs DDL

CREATE TABLE IF NOT EXISTS metapath_random_walk_schemas (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    graph_dataset_id VARCHAR(36) NOT NULL,
    metapath_sequence VARCHAR(100)[] NOT NULL,
    walk_length INT DEFAULT 20,
    num_walks_per_node INT DEFAULT 10,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
