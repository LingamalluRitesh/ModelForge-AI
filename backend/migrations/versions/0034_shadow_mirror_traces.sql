-- ModelForge AI - PostgreSQL Migration 0034: Dark Shadow Traffic Live Trace Samples DDL

CREATE TABLE IF NOT EXISTS shadow_traffic_traces (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(36) NOT NULL REFERENCES shadow_traffic_sessions(id) ON DELETE CASCADE,
    request_id VARCHAR(100) NOT NULL,
    primary_prediction FLOAT NOT NULL,
    shadow_prediction FLOAT NOT NULL,
    absolute_divergence FLOAT NOT NULL,
    primary_latency_ms FLOAT NOT NULL,
    shadow_latency_ms FLOAT NOT NULL,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
