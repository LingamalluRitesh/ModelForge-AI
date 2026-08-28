-- ModelForge AI - PostgreSQL Migration 0093: Triplet Metric Learning Vector Anchors DDL

CREATE TABLE IF NOT EXISTS triplet_metric_anchors (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    knowledge_base_id VARCHAR(36) NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    anchor_vector vector(512) NOT NULL,
    positive_cluster_id INT NOT NULL,
    hard_negative_min_distance FLOAT NOT NULL,
    indexed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
