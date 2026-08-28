-- ModelForge AI - PostgreSQL Migration 0023: Model Watermarking Verification DDL

CREATE TABLE IF NOT EXISTS model_watermarks (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version_id VARCHAR(36) NOT NULL REFERENCES model_versions(id) ON DELETE CASCADE,
    watermark_key_hash VARCHAR(100) NOT NULL,
    trigger_set_size INT DEFAULT 100,
    verification_success_rate FLOAT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
