-- ModelForge AI - PostgreSQL Migration 0057: Hindsight Experience Replay Storage DDL

CREATE TABLE IF NOT EXISTS hindsight_replay_episodes (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id VARCHAR(36) NOT NULL,
    episode_length INT NOT NULL,
    cumulative_reward FLOAT NOT NULL,
    transitions_count INT NOT NULL,
    storage_s3_path VARCHAR(500) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
