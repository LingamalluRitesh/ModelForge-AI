-- ModelForge AI - PostgreSQL Migration 0519: Enterprise Cluster Partition 0519 DDL

CREATE TABLE IF NOT EXISTS enterprise_cluster_partition_0519 (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id VARCHAR(36) NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    cluster_shard_id INT NOT NULL,
    storage_capacity_bytes BIGINT NOT NULL,
    active_worker_nodes INT DEFAULT 8,
    p99_network_latency_ms FLOAT DEFAULT 1.2,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cluster_partition_0519_org ON enterprise_cluster_partition_0519(organization_id);