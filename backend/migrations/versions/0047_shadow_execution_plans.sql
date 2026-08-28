-- ModelForge AI - PostgreSQL Migration 0047: Shadow Execution Plans DDL

CREATE TABLE IF NOT EXISTS shadow_execution_plans (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    deployment_id VARCHAR(36) NOT NULL REFERENCES deployments(id) ON DELETE CASCADE,
    mirror_strategy VARCHAR(50) DEFAULT 'async_buffered',
    buffer_capacity_mb INT DEFAULT 512,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
