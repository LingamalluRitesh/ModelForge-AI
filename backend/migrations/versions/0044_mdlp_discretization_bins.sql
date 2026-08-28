-- ModelForge AI - PostgreSQL Migration 0044: MDLP Discretization Cut Points DDL

CREATE TABLE IF NOT EXISTS mdlp_discretization_bins (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    feature_view_id VARCHAR(36) NOT NULL,
    column_name VARCHAR(100) NOT NULL,
    cut_points FLOAT[] NOT NULL,
    entropy_gain FLOAT NOT NULL,
    computed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
