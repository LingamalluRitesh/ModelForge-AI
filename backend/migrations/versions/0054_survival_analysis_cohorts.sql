-- ModelForge AI - PostgreSQL Migration 0054: Survival Analysis Cohort Curves DDL

CREATE TABLE IF NOT EXISTS survival_analysis_cohorts (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id VARCHAR(36) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    cohort_name VARCHAR(100) NOT NULL,
    c_index FLOAT NOT NULL,
    time_points_days INT[] NOT NULL,
    survival_probabilities FLOAT[] NOT NULL,
    computed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
