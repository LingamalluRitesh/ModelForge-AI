-- ModelForge AI - PostgreSQL Migration 0098: Portfolio Value-at-Risk Calculations DDL

CREATE TABLE IF NOT EXISTS model_portfolio_var_cvar_logs (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version_id VARCHAR(36) NOT NULL REFERENCES model_versions(id) ON DELETE CASCADE,
    confidence_level FLOAT DEFAULT 0.95,
    value_at_risk FLOAT NOT NULL,
    expected_shortfall FLOAT NOT NULL,
    worst_case_drawdown FLOAT NOT NULL,
    calculated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
