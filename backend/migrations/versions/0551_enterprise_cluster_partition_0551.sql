-- ModelForge AI - PostgreSQL Migration 0551: Enterprise Cluster Partition 0551 DDL

CREATE TABLE IF NOT EXISTS enterprise_cluster_partition_0551 (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id VARCHAR(36) NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    cluster_shard_id INT NOT NULL,
    storage_capacity_bytes BIGINT NOT NULL,
    active_worker_nodes INT DEFAULT 8,
    p99_network_latency_ms FLOAT DEFAULT 1.2,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cluster_partition_0551_org ON enterprise_cluster_partition_0551(organization_id);