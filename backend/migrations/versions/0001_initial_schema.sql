-- ModelForge AI - PostgreSQL Migration 0001: Initial Core Schema
-- Sets up Users, Organizations, Projects, Datasets, Experiments, Models, and Deployments.

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Organizations
CREATE TABLE IF NOT EXISTS organizations (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(150) NOT NULL,
    slug VARCHAR(100) UNIQUE NOT NULL,
    plan VARCHAR(50) DEFAULT 'developer',
    max_projects INT DEFAULT 20,
    max_storage_gb INT DEFAULT 100,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Users
CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    avatar_url VARCHAR(500),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Roles
CREATE TABLE IF NOT EXISTS roles (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(50) UNIQUE NOT NULL,
    display_name VARCHAR(100) NOT NULL,
    description TEXT,
    permissions JSONB DEFAULT '[]'::jsonb,
    is_system_role BOOLEAN DEFAULT FALSE
);

-- Organization Members
CREATE TABLE IF NOT EXISTS organization_members (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id VARCHAR(36) NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    user_id VARCHAR(36) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role_id VARCHAR(36) NOT NULL REFERENCES roles(id),
    is_owner BOOLEAN DEFAULT FALSE,
    joined_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT uq_org_user UNIQUE (organization_id, user_id)
);

-- Projects
CREATE TABLE IF NOT EXISTS projects (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id VARCHAR(36) NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    name VARCHAR(150) NOT NULL,
    slug VARCHAR(100) NOT NULL,
    description TEXT,
    problem_type VARCHAR(50) NOT NULL,
    tags JSONB DEFAULT '[]'::jsonb,
    is_archived BOOLEAN DEFAULT FALSE,
    created_by_id VARCHAR(36) NOT NULL REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT uq_org_proj_slug UNIQUE (organization_id, slug)
);

-- Datasets
CREATE TABLE IF NOT EXISTS datasets (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id VARCHAR(36) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name VARCHAR(150) NOT NULL,
    version VARCHAR(50) DEFAULT 'v1',
    description TEXT,
    storage_uri VARCHAR(500) NOT NULL,
    row_count INT NOT NULL,
    column_count INT NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    file_format VARCHAR(20) NOT NULL,
    schema_definition JSONB NOT NULL,
    statistics JSONB DEFAULT '{}'::jsonb,
    quality_score FLOAT,
    data_hash VARCHAR(64) NOT NULL,
    created_by_id VARCHAR(36) NOT NULL REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Registered Models
CREATE TABLE IF NOT EXISTS registered_models (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id VARCHAR(36) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name VARCHAR(150) NOT NULL,
    description TEXT,
    problem_type VARCHAR(50) NOT NULL,
    tags JSONB DEFAULT '[]'::jsonb,
    is_archived BOOLEAN DEFAULT FALSE,
    created_by_id VARCHAR(36) NOT NULL REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Model Versions
CREATE TABLE IF NOT EXISTS model_versions (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    registered_model_id VARCHAR(36) NOT NULL REFERENCES registered_models(id) ON DELETE CASCADE,
    version_tag VARCHAR(50) NOT NULL,
    stage VARCHAR(50) DEFAULT 'development',
    experiment_run_id VARCHAR(36),
    algorithm_name VARCHAR(100) NOT NULL,
    framework VARCHAR(50) NOT NULL,
    storage_uri VARCHAR(500) NOT NULL,
    signature JSONB DEFAULT '{}'::jsonb,
    metrics JSONB NOT NULL,
    hyperparameters JSONB DEFAULT '{}'::jsonb,
    quality_gate_passed BOOLEAN DEFAULT FALSE,
    quality_gate_summary JSONB,
    description TEXT,
    created_by_id VARCHAR(36) NOT NULL REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Deployments
CREATE TABLE IF NOT EXISTS deployments (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id VARCHAR(36) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name VARCHAR(150) NOT NULL,
    endpoint_path VARCHAR(100) UNIQUE NOT NULL,
    environment VARCHAR(50) DEFAULT 'production',
    status VARCHAR(50) DEFAULT 'active',
    model_version_id VARCHAR(36) NOT NULL REFERENCES model_versions(id),
    strategy VARCHAR(50) DEFAULT 'direct',
    secondary_model_version_id VARCHAR(36) REFERENCES model_versions(id),
    primary_traffic_percentage FLOAT DEFAULT 100.0,
    canary_stage_percentage FLOAT DEFAULT 0.0,
    min_replicas INT DEFAULT 1,
    max_replicas INT DEFAULT 5,
    current_replicas INT DEFAULT 1,
    cpu_limit VARCHAR(20) DEFAULT '1000m',
    memory_limit VARCHAR(20) DEFAULT '2Gi',
    is_healthy BOOLEAN DEFAULT TRUE,
    error_rate_threshold FLOAT DEFAULT 0.05,
    latency_threshold_ms FLOAT DEFAULT 500.0,
    auto_rollback_enabled BOOLEAN DEFAULT TRUE,
    created_by_id VARCHAR(36) NOT NULL REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for ultra-low query latency
CREATE INDEX IF NOT EXISTS idx_deployments_endpoint ON deployments(endpoint_path);
CREATE INDEX IF NOT EXISTS idx_model_versions_reg ON model_versions(registered_model_id);
CREATE INDEX IF NOT EXISTS idx_datasets_project ON datasets(project_id);
