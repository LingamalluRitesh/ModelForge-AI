-- ModelForge AI - PostgreSQL Migration 0004: Streaming Windows & Benchmark Telemetry DDL

-- Streaming Telemetry Windows
CREATE TABLE IF NOT EXISTS streaming_telemetry_windows (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    deployment_id VARCHAR(36) NOT NULL REFERENCES deployments(id) ON DELETE CASCADE,
    window_start TIMESTAMP WITH TIME ZONE NOT NULL,
    window_end TIMESTAMP WITH TIME ZONE NOT NULL,
    request_count BIGINT DEFAULT 0,
    p50_latency_ms FLOAT,
    p90_latency_ms FLOAT,
    p95_latency_ms FLOAT,
    p99_latency_ms FLOAT,
    psi_drift_score FLOAT,
    ks_statistic FLOAT,
    status VARCHAR(50) DEFAULT 'HEALTHY',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Model Benchmark Runs
CREATE TABLE IF NOT EXISTS model_benchmark_runs (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id VARCHAR(36) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name VARCHAR(150) NOT NULL,
    dataset_id VARCHAR(36) NOT NULL REFERENCES datasets(id),
    status VARCHAR(50) DEFAULT 'completed',
    leaderboard_results JSONB NOT NULL,
    winning_model_id VARCHAR(36),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
