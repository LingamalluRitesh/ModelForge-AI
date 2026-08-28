-- ModelForge AI - PostgreSQL Migration 0021: Complex Event Processing Rules DDL

CREATE TABLE IF NOT EXISTS streaming_cep_rules (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id VARCHAR(36) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name VARCHAR(150) NOT NULL,
    pattern_specification JSONB NOT NULL,
    time_window_seconds FLOAT DEFAULT 60.0,
    action_webhook_url VARCHAR(500),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
