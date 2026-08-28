-- ModelForge AI - PostgreSQL Migration 0006: Vector Embeddings, RAG & Semantic Search DDL

CREATE EXTENSION IF NOT EXISTS vector;

-- Knowledge Base Collections
CREATE TABLE IF NOT EXISTS knowledge_bases (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id VARCHAR(36) NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    name VARCHAR(150) NOT NULL,
    description TEXT,
    embedding_model VARCHAR(100) DEFAULT 'text-embedding-3-small',
    embedding_dimension INT DEFAULT 1536,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Document Chunks & Vector Store
CREATE TABLE IF NOT EXISTS document_chunks (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    knowledge_base_id VARCHAR(36) NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    document_name VARCHAR(255) NOT NULL,
    chunk_index INT NOT NULL,
    chunk_content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding vector(1536),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- HNSW Vector Index for Cosine Distance ANN Search
CREATE INDEX IF NOT EXISTS idx_document_chunks_hnsw 
ON document_chunks 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 100);

-- Query Analytics
CREATE TABLE IF NOT EXISTS vector_query_logs (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    knowledge_base_id VARCHAR(36) NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    query_text TEXT NOT NULL,
    top_k INT DEFAULT 5,
    latency_ms FLOAT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
