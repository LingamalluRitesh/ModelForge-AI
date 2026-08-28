-- ModelForge AI - PostgreSQL Migration 0058: Cluster-GCN Subgraph Partitions DDL

CREATE TABLE IF NOT EXISTS cluster_gcn_subgraphs (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    graph_dataset_id VARCHAR(36) NOT NULL,
    cluster_index INT NOT NULL,
    node_count INT NOT NULL,
    edge_count INT NOT NULL,
    density FLOAT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
