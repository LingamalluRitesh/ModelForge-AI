-- ModelForge AI - PostgreSQL Migration 0088: Continuous A/B Experiment Cohorts DDL

CREATE TABLE IF NOT EXISTS continuous_ab_experiment_cohorts (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    deployment_id VARCHAR(36) NOT NULL REFERENCES deployments(id) ON DELETE CASCADE,
    variant_name VARCHAR(50) NOT NULL,
    traffic_percentage FLOAT NOT NULL,
    conversion_count BIGINT DEFAULT 0,
    total_exposure_count BIGINT DEFAULT 0,
    is_winner BOOLEAN DEFAULT FALSE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
