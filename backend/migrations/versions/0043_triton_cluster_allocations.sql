-- ModelForge AI - PostgreSQL Migration 0043: Triton GPU Server Allocations DDL

CREATE TABLE IF NOT EXISTS triton_server_instances (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    deployment_id VARCHAR(36) NOT NULL REFERENCES deployments(id) ON DELETE CASCADE,
    gpu_device_id VARCHAR(50) NOT NULL,
    gpu_memory_allocated_mb INT DEFAULT 16384,
    instance_ip VARCHAR(50) NOT NULL,
    health_status VARCHAR(50) DEFAULT 'HEALTHY',
    last_heartbeat_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
