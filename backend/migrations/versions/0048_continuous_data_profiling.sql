-- ModelForge AI - PostgreSQL Migration 0048: Continuous Data Profiling Histograms DDL

CREATE TABLE IF NOT EXISTS streaming_feature_histograms (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    feature_view_id VARCHAR(36) NOT NULL,
    window_start TIMESTAMP WITH TIME ZONE NOT NULL,
    window_end TIMESTAMP WITH TIME ZONE NOT NULL,
    histogram_bins JSONB NOT NULL,
    quantile_p50 FLOAT NOT NULL,
    quantile_p99 FLOAT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
