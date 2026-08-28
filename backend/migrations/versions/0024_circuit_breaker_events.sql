-- ModelForge AI - PostgreSQL Migration 0024: Serving Circuit Breaker Events DDL

CREATE TABLE IF NOT EXISTS circuit_breaker_events (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    deployment_id VARCHAR(36) NOT NULL REFERENCES deployments(id) ON DELETE CASCADE,
    state_from VARCHAR(50) NOT NULL,
    state_to VARCHAR(50) NOT NULL,
    reason TEXT NOT NULL,
    tripped_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
