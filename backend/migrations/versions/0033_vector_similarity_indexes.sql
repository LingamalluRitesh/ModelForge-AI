-- ModelForge AI - PostgreSQL Migration 0033: Multi-Index Vector Indexing Configurations DDL

CREATE TABLE IF NOT EXISTS vector_index_configs (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    knowledge_base_id VARCHAR(36) NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    index_type VARCHAR(50) DEFAULT 'hnsw', -- hnsw, ivfflat, scann
    metric VARCHAR(50) DEFAULT 'cosine',
    hnsw_m INT DEFAULT 16,
    hnsw_ef_construction INT DEFAULT 100,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
