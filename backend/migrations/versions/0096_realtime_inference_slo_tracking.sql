-- ModelForge AI - PostgreSQL Migration 0096: Real-Time Inference SLO Tracking DDL

CREATE TABLE IF NOT EXISTS real_time_inference_slo_tracking (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    deployment_id VARCHAR(36) NOT NULL REFERENCES deployments(id) ON DELETE CASCADE,
    minute_bucket TIMESTAMP WITH TIME ZONE NOT NULL,
    total_requests INT NOT NULL,
    sub_10ms_count INT NOT NULL,
    p99_latency_ms FLOAT NOT NULL,
    p95_latency_ms FLOAT NOT NULL,
    error_count INT DEFAULT 0,
    slo_compliance_pct FLOAT NOT NULL,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
