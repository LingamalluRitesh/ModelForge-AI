-- ModelForge AI - PostgreSQL Migration 0035: Distributed Checkpoint Manifests DDL

CREATE TABLE IF NOT EXISTS distributed_checkpoint_manifests (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id VARCHAR(36) NOT NULL REFERENCES distributed_training_jobs(id) ON DELETE CASCADE,
    epoch INT NOT NULL,
    step INT NOT NULL,
    storage_s3_uri VARCHAR(500) NOT NULL,
    loss FLOAT NOT NULL,
    saved_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
