-- ModelForge AI - PostgreSQL Migration 0029: Streaming Matrix Profiles DDL

CREATE TABLE IF NOT EXISTS streaming_matrix_profiles (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    deployment_id VARCHAR(36) NOT NULL REFERENCES deployments(id) ON DELETE CASCADE,
    window_size INT NOT NULL,
    min_discord_distance FLOAT NOT NULL,
    max_discord_distance FLOAT NOT NULL,
    anomaly_detected BOOLEAN DEFAULT FALSE,
    computed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
