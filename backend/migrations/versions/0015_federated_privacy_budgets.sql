-- ModelForge AI - PostgreSQL Migration 0015: Differential Privacy Accounting DDL

CREATE TABLE IF NOT EXISTS federated_privacy_budgets (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(36) UNIQUE NOT NULL REFERENCES federated_sessions(id) ON DELETE CASCADE,
    target_epsilon FLOAT NOT NULL,
    target_delta FLOAT NOT NULL,
    consumed_epsilon FLOAT DEFAULT 0.0,
    consumed_delta FLOAT DEFAULT 0.0,
    is_exhausted BOOLEAN DEFAULT FALSE,
    last_computed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
