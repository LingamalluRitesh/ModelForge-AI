-- ModelForge AI - PostgreSQL Migration 0008: Synthetic Data Catalog & Privacy Audits DDL

-- Synthetic Data Generation Jobs
CREATE TABLE IF NOT EXISTS synthetic_dataset_jobs (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id VARCHAR(36) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name VARCHAR(150) NOT NULL,
    source_dataset_id VARCHAR(36) NOT NULL REFERENCES datasets(id),
    generator_algorithm VARCHAR(50) NOT NULL, -- ctgan, gaussian_copula, privbayes
    num_samples_generated INT NOT NULL,
    storage_uri VARCHAR(500) NOT NULL,
    fidelity_score FLOAT,
    privacy_score FLOAT,
    differential_privacy_epsilon FLOAT,
    differential_privacy_delta FLOAT,
    status VARCHAR(50) DEFAULT 'COMPLETED',
    parameters JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Privacy Attack Evaluations
CREATE TABLE IF NOT EXISTS privacy_audit_evaluations (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    synthetic_dataset_job_id VARCHAR(36) NOT NULL REFERENCES synthetic_dataset_jobs(id) ON DELETE CASCADE,
    attack_type VARCHAR(100) NOT NULL, -- membership_inference, attribute_disclosure, shadow_model
    risk_score FLOAT NOT NULL,
    empirical_privacy_passed BOOLEAN DEFAULT TRUE,
    attack_metrics JSONB NOT NULL,
    evaluated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
