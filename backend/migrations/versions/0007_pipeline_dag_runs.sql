-- ModelForge AI - PostgreSQL Migration 0007: Pipeline DAG Runs & Task State DDL

-- Pipelines
CREATE TABLE IF NOT EXISTS ml_pipelines (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id VARCHAR(36) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name VARCHAR(150) NOT NULL,
    description TEXT,
    schedule_cron VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    dag_definition JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Pipeline Runs
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    pipeline_id VARCHAR(36) NOT NULL REFERENCES ml_pipelines(id) ON DELETE CASCADE,
    trigger_type VARCHAR(50) DEFAULT 'manual', -- manual, scheduled, drift_triggered
    status VARCHAR(50) DEFAULT 'RUNNING',      -- RUNNING, COMPLETED, FAILED, CANCELLED
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    finished_at TIMESTAMP WITH TIME ZONE,
    duration_seconds FLOAT,
    error_message TEXT,
    execution_context JSONB DEFAULT '{}'::jsonb
);

-- Task Runs
CREATE TABLE IF NOT EXISTS pipeline_task_runs (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    pipeline_run_id VARCHAR(36) NOT NULL REFERENCES pipeline_runs(id) ON DELETE CASCADE,
    task_id VARCHAR(100) NOT NULL,
    task_name VARCHAR(150) NOT NULL,
    status VARCHAR(50) DEFAULT 'PENDING',
    retry_count INT DEFAULT 0,
    started_at TIMESTAMP WITH TIME ZONE,
    finished_at TIMESTAMP WITH TIME ZONE,
    duration_seconds FLOAT,
    inputs JSONB,
    outputs JSONB,
    logs TEXT,
    error_message TEXT
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_pipeline ON pipeline_runs(pipeline_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_task_runs_run ON pipeline_task_runs(pipeline_run_id);
