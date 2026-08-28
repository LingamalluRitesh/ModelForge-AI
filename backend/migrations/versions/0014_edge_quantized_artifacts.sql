-- ModelForge AI - PostgreSQL Migration 0014: Compiled & Quantized Edge Artifacts DDL

CREATE TABLE IF NOT EXISTS compiled_edge_artifacts (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version_id VARCHAR(36) NOT NULL REFERENCES model_versions(id) ON DELETE CASCADE,
    target_backend VARCHAR(50) NOT NULL, -- onnx_int8, tensorrt_fp16, openvino, coreml
    optimization_level INT DEFAULT 3,
    original_size_bytes BIGINT NOT NULL,
    compiled_size_bytes BIGINT NOT NULL,
    compression_ratio FLOAT NOT NULL,
    p99_latency_reduction_pct FLOAT NOT NULL,
    storage_uri VARCHAR(500) NOT NULL,
    compiled_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
