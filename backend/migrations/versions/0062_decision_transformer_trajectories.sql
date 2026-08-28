-- ModelForge AI - PostgreSQL Migration 0062: Decision Transformer Offline Rollouts DDL

CREATE TABLE IF NOT EXISTS decision_transformer_trajectories (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id VARCHAR(36) NOT NULL,
    trajectory_length INT NOT NULL,
    target_return FLOAT NOT NULL,
    achieved_return FLOAT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
