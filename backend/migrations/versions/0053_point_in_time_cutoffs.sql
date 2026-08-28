-- ModelForge AI - PostgreSQL Migration 0053: Point-in-Time Join Cutoff Logs DDL

CREATE TABLE IF NOT EXISTS point_in_time_join_logs (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id VARCHAR(36) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    observation_count BIGINT NOT NULL,
    feature_count INT NOT NULL,
    execution_duration_ms FLOAT NOT NULL,
    joined_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
