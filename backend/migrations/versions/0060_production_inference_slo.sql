-- ModelForge AI - PostgreSQL Migration 0060: Production Inference SLO & Error Budgets DDL

CREATE TABLE IF NOT EXISTS production_inference_slo_metrics (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    deployment_id VARCHAR(36) NOT NULL REFERENCES deployments(id) ON DELETE CASCADE,
    target_p99_latency_ms FLOAT DEFAULT 10.0,
    actual_p99_latency_ms FLOAT NOT NULL,
    target_availability_pct FLOAT DEFAULT 99.99,
    actual_availability_pct FLOAT NOT NULL,
    error_budget_remaining_pct FLOAT NOT NULL,
    measured_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
