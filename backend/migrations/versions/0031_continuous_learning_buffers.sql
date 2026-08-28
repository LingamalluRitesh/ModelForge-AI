-- ModelForge AI - PostgreSQL Migration 0031: Continuous Online Learning Replay Buffers DDL

CREATE TABLE IF NOT EXISTS online_replay_buffers (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    deployment_id VARCHAR(36) NOT NULL REFERENCES deployments(id) ON DELETE CASCADE,
    buffer_capacity INT DEFAULT 50000,
    current_size INT DEFAULT 0,
    priority_alpha FLOAT DEFAULT 0.6,
    storage_s3_path VARCHAR(500) NOT NULL,
    last_eviction_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
