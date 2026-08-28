-- ModelForge AI - PostgreSQL Migration 0061: High-Throughput Redis Online Feature Cache DDL

CREATE TABLE IF NOT EXISTS online_feature_cache_metadata (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    feature_view_id VARCHAR(36) NOT NULL,
    redis_cluster_uri VARCHAR(500) NOT NULL,
    cache_ttl_seconds INT DEFAULT 86400,
    hit_rate_pct FLOAT DEFAULT 99.4,
    eviction_policy VARCHAR(50) DEFAULT 'volatile-lru',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
