-- ModelForge AI - PostgreSQL Migration 0087: Online Feature Transformation Code Repository DDL

CREATE TABLE IF NOT EXISTS online_feature_transformations (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    feature_view_id VARCHAR(36) NOT NULL,
    python_udf_code TEXT NOT NULL,
    compiled_ast_hash VARCHAR(64) NOT NULL,
    is_sandboxed BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
