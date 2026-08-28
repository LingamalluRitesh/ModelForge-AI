-- ModelForge AI - PostgreSQL Migration 0066: Neural Collaborative Filtering Models DDL

CREATE TABLE IF NOT EXISTS ncf_recommendation_models (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id VARCHAR(36) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    num_users INT NOT NULL,
    num_items INT NOT NULL,
    hit_ratio_at_10 FLOAT NOT NULL,
    ndcg_at_10 FLOAT NOT NULL,
    storage_weights_uri VARCHAR(500) NOT NULL,
    trained_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
