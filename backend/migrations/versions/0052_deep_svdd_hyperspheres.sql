-- ModelForge AI - PostgreSQL Migration 0052: Deep SVDD Hypersphere Parameters DDL

CREATE TABLE IF NOT EXISTS deep_svdd_hyperspheres (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version_id VARCHAR(36) NOT NULL REFERENCES model_versions(id) ON DELETE CASCADE,
    center_vector FLOAT[] NOT NULL,
    radius_squared FLOAT NOT NULL,
    nu_parameter FLOAT DEFAULT 0.05,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
