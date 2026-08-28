-- ModelForge AI - PostgreSQL Migration 0012: Dark Shadow Traffic Telemetry DDL

CREATE TABLE IF NOT EXISTS shadow_traffic_sessions (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    deployment_id VARCHAR(36) NOT NULL REFERENCES deployments(id) ON DELETE CASCADE,
    primary_model_version_id VARCHAR(36) NOT NULL REFERENCES model_versions(id),
    shadow_model_version_id VARCHAR(36) NOT NULL REFERENCES model_versions(id),
    sampling_percentage FLOAT DEFAULT 100.0,
    total_mirrored_requests BIGINT DEFAULT 0,
    discrepancy_count BIGINT DEFAULT 0,
    primary_p99_latency_ms FLOAT,
    shadow_p99_latency_ms FLOAT,
    status VARCHAR(50) DEFAULT 'ACTIVE',
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    stopped_at TIMESTAMP WITH TIME ZONE
);
