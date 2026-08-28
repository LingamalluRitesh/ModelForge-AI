-- ModelForge AI - PostgreSQL Migration 0019: OpenLineage Run & Dataset Facets DDL

CREATE TABLE IF NOT EXISTS openlineage_run_events (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id VARCHAR(36) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    event_type VARCHAR(50) NOT NULL, -- START, RUNNING, COMPLETE, ABORT, FAIL
    job_namespace VARCHAR(100) NOT NULL,
    job_name VARCHAR(150) NOT NULL,
    run_id VARCHAR(36) NOT NULL,
    inputs JSONB,
    outputs JSONB,
    producer VARCHAR(100) DEFAULT 'modelforge-openlineage-v1',
    event_time TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
