-- ModelForge AI - PostgreSQL Migration 0067: Multi-Agent RL Training Sessions DDL

CREATE TABLE IF NOT EXISTS multi_agent_rl_sessions (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id VARCHAR(36) NOT NULL,
    num_active_agents INT NOT NULL,
    environment_type VARCHAR(100) NOT NULL,
    cooperative_reward_sum FLOAT NOT NULL,
    competitive_entropy FLOAT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
