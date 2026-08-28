-- ModelForge AI - PostgreSQL Migration 0042: Secure Aggregation Sessions DDL

CREATE TABLE IF NOT EXISTS secure_aggregation_sessions (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    federated_round_id VARCHAR(36) NOT NULL,
    protocol_version VARCHAR(20) DEFAULT 'bonawitz_secagg_v2',
    participating_clients_count INT NOT NULL,
    status VARCHAR(50) DEFAULT 'COMPLETED',
    aggregated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
