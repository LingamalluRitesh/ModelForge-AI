-- ModelForge AI - PostgreSQL Migration 0094: Rényi DP Privacy Composition Logs DDL

CREATE TABLE IF NOT EXISTS renyi_dp_composition_logs (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    federated_session_id VARCHAR(36) NOT NULL,
    subsample_ratio FLOAT NOT NULL,
    gaussian_sigma FLOAT NOT NULL,
    computed_epsilon FLOAT NOT NULL,
    target_delta FLOAT DEFAULT 0.00001,
    logged_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
