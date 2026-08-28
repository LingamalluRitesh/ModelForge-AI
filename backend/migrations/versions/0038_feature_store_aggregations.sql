-- ModelForge AI - PostgreSQL Migration 0038: Feature Store Materialization Views DDL

CREATE TABLE IF NOT EXISTS materialized_feature_views (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    feature_view_name VARCHAR(100) UNIQUE NOT NULL,
    entity_key VARCHAR(50) NOT NULL,
    aggregation_window VARCHAR(20) NOT NULL, -- 1h, 24h, 7d, 30d
    last_materialized_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    storage_engine VARCHAR(50) DEFAULT 'redis_and_delta'
);
