-- ModelForge AI - PostgreSQL Migration 0039: Distributed Parameter Server Checkpoints DDL

CREATE TABLE IF NOT EXISTS parameter_server_checkpoints (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id VARCHAR(36) NOT NULL REFERENCES distributed_training_jobs(id) ON DELETE CASCADE,
    shard_id INT NOT NULL,
    storage_s3_key VARCHAR(500) NOT NULL,
    checksum_sha256 VARCHAR(64) NOT NULL,
    committed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
