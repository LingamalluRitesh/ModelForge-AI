-- ModelForge AI - PostgreSQL Migration 0078: Multi-Region Model Replication Topology DDL

CREATE TABLE IF NOT EXISTS multi_region_replication_topology (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version_id VARCHAR(36) NOT NULL REFERENCES model_versions(id) ON DELETE CASCADE,
    primary_region VARCHAR(50) NOT NULL,
    replicated_regions VARCHAR(50)[] NOT NULL,
    replication_status VARCHAR(50) DEFAULT 'SYNCHRONIZED',
    last_synced_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
