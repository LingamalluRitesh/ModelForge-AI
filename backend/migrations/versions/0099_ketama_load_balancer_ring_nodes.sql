-- ModelForge AI - PostgreSQL Migration 0099: Ketama Load Balancer Virtual Ring Nodes DDL

CREATE TABLE IF NOT EXISTS ketama_load_balancer_ring_nodes (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    cluster_name VARCHAR(100) NOT NULL,
    node_id VARCHAR(100) NOT NULL,
    virtual_point_hashes INT[] NOT NULL,
    registered_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
