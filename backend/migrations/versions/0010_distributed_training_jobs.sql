-- ModelForge AI - PostgreSQL Migration 0010: Distributed Ray / PyTorch DDP Training Jobs DDL

-- Distributed Training Jobs
CREATE TABLE IF NOT EXISTS distributed_training_jobs (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id VARCHAR(36) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name VARCHAR(150) NOT NULL,
    framework VARCHAR(50) NOT NULL, -- ray_train, pytorch_ddp, deepspeed, horovod
    num_nodes INT DEFAULT 1,
    gpus_per_node INT DEFAULT 1,
    cluster_type VARCHAR(50) DEFAULT 'k8s_ray_cluster',
    job_status VARCHAR(50) DEFAULT 'INITIALIZING', -- INITIALIZING, RUNNING, SUCCEEDED, FAILED
    entrypoint_command TEXT NOT NULL,
    hyperparameters JSONB DEFAULT '{}'::jsonb,
    final_metrics JSONB,
    checkpoint_storage_uri VARCHAR(500),
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    created_by_id VARCHAR(36) NOT NULL REFERENCES users(id)
);

-- Distributed Node Metric Snapshots
CREATE TABLE IF NOT EXISTS distributed_node_metrics (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id VARCHAR(36) NOT NULL REFERENCES distributed_training_jobs(id) ON DELETE CASCADE,
    node_id VARCHAR(100) NOT NULL,
    step_number INT NOT NULL,
    gpu_utilization_pct FLOAT,
    gpu_memory_used_mb FLOAT,
    gradient_allreduce_latency_ms FLOAT,
    throughput_samples_per_sec FLOAT,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_dist_jobs_proj ON distributed_training_jobs(project_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_dist_metrics_job ON distributed_node_metrics(job_id, step_number);
