-- ModelForge AI - PostgreSQL Migration 0013: Causal Uplift & Qini Curves DDL

CREATE TABLE IF NOT EXISTS causal_uplift_cohorts (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    causal_experiment_id VARCHAR(36) NOT NULL REFERENCES causal_experiments(id) ON DELETE CASCADE,
    decile INT NOT NULL,
    treatment_sample_count INT NOT NULL,
    control_sample_count INT NOT NULL,
    treatment_conversion_rate FLOAT NOT NULL,
    control_conversion_rate FLOAT NOT NULL,
    incremental_uplift FLOAT NOT NULL,
    cumulative_qini_score FLOAT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
