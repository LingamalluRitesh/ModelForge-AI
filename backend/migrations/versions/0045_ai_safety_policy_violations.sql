-- ModelForge AI - PostgreSQL Migration 0045: AI Safety Guardrail Violations DDL

CREATE TABLE IF NOT EXISTS ai_safety_policy_violations (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    deployment_id VARCHAR(36) NOT NULL REFERENCES deployments(id) ON DELETE CASCADE,
    rule_name VARCHAR(100) NOT NULL,
    violation_category VARCHAR(50) NOT NULL,
    flagged_payload_snippet TEXT NOT NULL,
    remediated_action VARCHAR(50) NOT NULL,
    occurred_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
