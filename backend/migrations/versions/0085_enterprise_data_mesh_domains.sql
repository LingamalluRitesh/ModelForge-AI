-- ModelForge AI - PostgreSQL Migration 0085: Enterprise Data Mesh Domains DDL

CREATE TABLE IF NOT EXISTS enterprise_data_mesh_domains (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    domain_name VARCHAR(100) UNIQUE NOT NULL,
    domain_owner VARCHAR(150) NOT NULL,
    data_contract_json JSONB NOT NULL,
    sla_availability_pct FLOAT DEFAULT 99.95,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
