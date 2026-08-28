-- ModelForge AI - PostgreSQL Migration 0003: Causal Inference & Federated Learning DDL

-- Causal Inference Experiments
CREATE TABLE IF NOT EXISTS causal_experiments (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id VARCHAR(36) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name VARCHAR(150) NOT NULL,
    method VARCHAR(50) NOT NULL, -- dml, psm, s_learner, t_learner, x_learner
    treatment_column VARCHAR(100) NOT NULL,
    outcome_column VARCHAR(100) NOT NULL,
    confounders JSONB NOT NULL,
    estimated_ate FLOAT NOT NULL,
    ate_ci_lower FLOAT NOT NULL,
    ate_ci_upper FLOAT NOT NULL,
    p_value FLOAT,
    cate_summary JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Federated Learning Rounds & Privacy
CREATE TABLE IF NOT EXISTS federated_sessions (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id VARCHAR(36) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name VARCHAR(150) NOT NULL,
    algorithm VARCHAR(50) DEFAULT 'fedavg',
    total_rounds INT DEFAULT 10,
    current_round INT DEFAULT 0,
    min_clients_per_round INT DEFAULT 3,
    differential_privacy_epsilon FLOAT DEFAULT 1.0,
    differential_privacy_delta FLOAT DEFAULT 1e-5,
    status VARCHAR(50) DEFAULT 'active',
    global_model_weights_uri VARCHAR(500),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS federated_client_updates (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(36) NOT NULL REFERENCES federated_sessions(id) ON DELETE CASCADE,
    client_id VARCHAR(100) NOT NULL,
    round_number INT NOT NULL,
    num_samples INT NOT NULL,
    loss FLOAT,
    metrics JSONB,
    received_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
