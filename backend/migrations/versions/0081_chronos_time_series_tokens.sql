-- ModelForge AI - PostgreSQL Migration 0081: Chronos Token Vocabulary Catalogs DDL

CREATE TABLE IF NOT EXISTS chronos_token_catalogs (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version_id VARCHAR(36) NOT NULL REFERENCES model_versions(id) ON DELETE CASCADE,
    vocab_size INT DEFAULT 4096,
    mean_scaling_factor FLOAT NOT NULL,
    storage_tokenizer_uri VARCHAR(500) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
