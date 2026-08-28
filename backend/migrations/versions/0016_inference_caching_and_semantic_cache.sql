-- ModelForge AI - PostgreSQL Migration 0016: Semantic Inference Caching DDL

CREATE TABLE IF NOT EXISTS semantic_inference_cache (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    deployment_id VARCHAR(36) NOT NULL REFERENCES deployments(id) ON DELETE CASCADE,
    input_hash VARCHAR(64) UNIQUE NOT NULL,
    input_vector vector(1536),
    cached_prediction JSONB NOT NULL,
    hit_count BIGINT DEFAULT 1,
    last_accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cache_hash ON semantic_inference_cache(input_hash);
