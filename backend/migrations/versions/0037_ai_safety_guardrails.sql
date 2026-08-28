-- ModelForge AI - PostgreSQL Migration 0037: Real-Time AI Safety Guardrails DDL

CREATE TABLE IF NOT EXISTS ai_safety_guardrail_rules (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id VARCHAR(36) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    rule_name VARCHAR(100) NOT NULL,
    filter_category VARCHAR(50) NOT NULL, -- PII_LEAKAGE, TOXICITY, PROMPT_INJECTION, FACTUAL_HALLUCINATION
    action_policy VARCHAR(50) DEFAULT 'REDACT_AND_LOG',
    confidence_threshold FLOAT DEFAULT 0.85,
    is_enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
