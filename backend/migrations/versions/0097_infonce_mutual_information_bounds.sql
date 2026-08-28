-- ModelForge AI - PostgreSQL Migration 0097: InfoNCE Mutual Information Bounds DDL

CREATE TABLE IF NOT EXISTS infonce_representation_bounds (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id VARCHAR(36) NOT NULL,
    mutual_information_bound FLOAT NOT NULL,
    temperature FLOAT DEFAULT 0.07,
    negative_sample_count INT DEFAULT 256,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
