-- ModelForge AI - PostgreSQL Migration 0018: AutoML NAS Architecture Search DDL

CREATE TABLE IF NOT EXISTS automl_nas_trials (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id VARCHAR(36) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    trial_number INT NOT NULL,
    architecture_spec JSONB NOT NULL,
    val_accuracy FLOAT NOT NULL,
    val_f1 FLOAT NOT NULL,
    parameter_count_millions FLOAT NOT NULL,
    inference_latency_ms FLOAT NOT NULL,
    is_pareto_optimal BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
