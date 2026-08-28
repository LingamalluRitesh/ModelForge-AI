-- ModelForge AI - PostgreSQL Migration 0002: Feature Store & Governance DDL

-- Feature Store Entities
CREATE TABLE IF NOT EXISTS feature_entities (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id VARCHAR(36) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    join_key VARCHAR(100) NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT uq_entity_project UNIQUE (project_id, name)
);

-- Feature Views
CREATE TABLE IF NOT EXISTS feature_views (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_id VARCHAR(36) NOT NULL REFERENCES feature_entities(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    features_schema JSONB NOT NULL,
    ttl_seconds INT DEFAULT 86400,
    online_storage_enabled BOOLEAN DEFAULT TRUE,
    offline_table_name VARCHAR(150),
    transformation_sql TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Governance Policies & Audits
CREATE TABLE IF NOT EXISTS model_approval_requests (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version_id VARCHAR(36) NOT NULL REFERENCES model_versions(id) ON DELETE CASCADE,
    target_stage VARCHAR(50) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    requested_by_id VARCHAR(36) NOT NULL REFERENCES users(id),
    reviewed_by_id VARCHAR(36) REFERENCES users(id),
    review_comments TEXT,
    quality_checks_passed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    reviewed_at TIMESTAMP WITH TIME ZONE
);

CREATE TABLE IF NOT EXISTS quality_gate_configs (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id VARCHAR(36) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    target_stage VARCHAR(50) DEFAULT 'production',
    rules JSONB NOT NULL,
    is_blocking BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS deployment_rollback_histories (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    deployment_id VARCHAR(36) NOT NULL REFERENCES deployments(id) ON DELETE CASCADE,
    from_model_version_id VARCHAR(36) NOT NULL,
    to_model_version_id VARCHAR(36) NOT NULL,
    trigger_type VARCHAR(50) DEFAULT 'manual',
    reason TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
