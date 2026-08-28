-- ModelForge AI - PostgreSQL Migration 0032: Knowledge Distillation Compression Lineage DDL

CREATE TABLE IF NOT EXISTS knowledge_distillation_runs (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id VARCHAR(36) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    teacher_model_version_id VARCHAR(36) NOT NULL REFERENCES model_versions(id),
    student_model_version_id VARCHAR(36) NOT NULL REFERENCES model_versions(id),
    temperature FLOAT DEFAULT 4.0,
    alpha_soft_loss FLOAT DEFAULT 0.7,
    compression_ratio FLOAT NOT NULL,
    latency_reduction_factor FLOAT NOT NULL,
    accuracy_retention_pct FLOAT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
