-- ModelForge AI - PostgreSQL Migration 0185: Enterprise Partition Shard 185 DDL

CREATE TABLE IF NOT EXISTS enterprise_partition_shard_185 (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id VARCHAR(36) NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    shard_sequence_id INT NOT NULL,
    storage_tier VARCHAR(50) DEFAULT 'HOT_NVME_TIER',
    allocated_gb INT DEFAULT 500,
    active_connections INT DEFAULT 24,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_shard_185_org ON enterprise_partition_shard_185(organization_id);
