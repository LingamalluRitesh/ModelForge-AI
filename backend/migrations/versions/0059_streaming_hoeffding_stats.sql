-- ModelForge AI - PostgreSQL Migration 0059: Streaming Hoeffding Tree Checkpoints DDL

CREATE TABLE IF NOT EXISTS hoeffding_tree_snapshots (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    deployment_id VARCHAR(36) NOT NULL REFERENCES deployments(id) ON DELETE CASCADE,
    samples_observed BIGINT NOT NULL,
    tree_depth INT NOT NULL,
    active_leaves INT NOT NULL,
    snapshot_json JSONB NOT NULL,
    saved_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
