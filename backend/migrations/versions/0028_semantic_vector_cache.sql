-- ModelForge AI - PostgreSQL Migration 0028: High-Throughput Semantic Vector Cache DDL

CREATE TABLE IF NOT EXISTS semantic_vector_query_cache (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    knowledge_base_id VARCHAR(36) NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    query_text_normalized TEXT NOT NULL,
    query_vector vector(1536) NOT NULL,
    response_payload JSONB NOT NULL,
    cache_hits BIGINT DEFAULT 1,
    last_hit_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_query_vec_cache ON semantic_vector_query_cache USING hnsw (query_vector vector_cosine_ops);
