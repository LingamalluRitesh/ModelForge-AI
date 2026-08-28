-- ModelForge AI - PostgreSQL Migration 0025: Feature Store Lineage Snapshots DDL

CREATE TABLE IF NOT EXISTS feature_store_lineage_snapshots (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id VARCHAR(36) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    feature_view_id VARCHAR(36) NOT NULL,
    snapshot_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    row_count BIGINT NOT NULL,
    column_schemas JSONB NOT NULL,
    storage_parquet_uri VARCHAR(500) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
