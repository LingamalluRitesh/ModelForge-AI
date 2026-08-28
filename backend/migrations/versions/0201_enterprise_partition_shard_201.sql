-- ModelForge AI - PostgreSQL Migration 0201: Enterprise Partition Shard 201 DDL

CREATE TABLE IF NOT EXISTS enterprise_partition_shard_201 (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id VARCHAR(36) NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    shard_sequence_id INT NOT NULL,
    storage_tier VARCHAR(50) DEFAULT 'HOT_NVME_TIER',
    allocated_gb INT DEFAULT 500,
    active_connections INT DEFAULT 24,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_shard_201_org ON enterprise_partition_shard_201(organization_id);
