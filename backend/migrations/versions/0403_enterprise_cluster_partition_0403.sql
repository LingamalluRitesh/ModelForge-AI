-- ModelForge AI - PostgreSQL Migration 0403: Enterprise Cluster Partition 0403 DDL

CREATE TABLE IF NOT EXISTS enterprise_cluster_partition_0403 (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id VARCHAR(36) NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    cluster_shard_id INT NOT NULL,
    storage_capacity_bytes BIGINT NOT NULL,
    active_worker_nodes INT DEFAULT 8,
    p99_network_latency_ms FLOAT DEFAULT 1.2,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cluster_partition_0403_org ON enterprise_cluster_partition_0403(organization_id);