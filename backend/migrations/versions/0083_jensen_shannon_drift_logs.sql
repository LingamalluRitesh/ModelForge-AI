-- ModelForge AI - PostgreSQL Migration 0083: Jensen-Shannon Divergence Telemetry Logs DDL

CREATE TABLE IF NOT EXISTS jensen_shannon_drift_logs (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    deployment_id VARCHAR(36) NOT NULL REFERENCES deployments(id) ON DELETE CASCADE,
    feature_name VARCHAR(100) NOT NULL,
    jsd_score FLOAT NOT NULL,
    threshold FLOAT DEFAULT 0.10,
    drift_alert_triggered BOOLEAN DEFAULT FALSE,
    evaluated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
