-- ModelForge AI - PostgreSQL Migration 0261: Real-Time Stream Joins and CEP State DDL

CREATE TABLE IF NOT EXISTS realtime_cep_stream_states (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id VARCHAR(36) NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    pattern_id VARCHAR(100) NOT NULL,
    current_state VARCHAR(50) NOT NULL,
    matched_events_count INT DEFAULT 0,
    window_start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    window_expiry_time TIMESTAMP WITH TIME ZONE NOT NULL,
    payload_snapshot JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cep_state_org_pattern ON realtime_cep_stream_states(organization_id, pattern_id);
