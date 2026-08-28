-- ModelForge AI - PostgreSQL Migration 0075: DANet Feature Abstract Extraction Configs DDL

CREATE TABLE IF NOT EXISTS danet_model_configurations (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version_id VARCHAR(36) NOT NULL REFERENCES model_versions(id) ON DELETE CASCADE,
    num_abstract_blocks INT DEFAULT 3,
    abstract_dimension INT DEFAULT 16,
    reconstruction_loss FLOAT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
