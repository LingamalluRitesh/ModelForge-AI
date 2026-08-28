-- ModelForge AI - PostgreSQL Migration 0104: Feather Streaming Checkpoints DDL

CREATE TABLE IF NOT EXISTS feather_streaming_partition_checkpoints (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    feature_view_id VARCHAR(36) NOT NULL,
    chunk_index INT NOT NULL,
    row_count INT NOT NULL,
    memory_mapped_offset BIGINT NOT NULL,
    committed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
