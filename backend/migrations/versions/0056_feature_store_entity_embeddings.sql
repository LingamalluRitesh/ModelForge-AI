-- ModelForge AI - PostgreSQL Migration 0056: Entity Embeddings Feature Table DDL

CREATE TABLE IF NOT EXISTS feature_entity_embeddings (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_key VARCHAR(100) NOT NULL,
    feature_group VARCHAR(100) NOT NULL,
    dense_vector vector(512) NOT NULL,
    last_updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_entity_emb ON feature_entity_embeddings USING ivfflat (dense_vector vector_cosine_ops);
