-- ModelForge AI - PostgreSQL Migration 0036: Distributed Ring AllReduce Telemetry DDL

CREATE TABLE IF NOT EXISTS ring_allreduce_telemetry (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id VARCHAR(36) NOT NULL REFERENCES distributed_training_jobs(id) ON DELETE CASCADE,
    step INT NOT NULL,
    bandwidth_gbps FLOAT NOT NULL,
    step_latency_ms FLOAT NOT NULL,
    gradient_norm FLOAT NOT NULL,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
