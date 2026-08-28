-- ModelForge AI - PostgreSQL Migration 0077: CloudWatch Alert Subscriptions DDL

CREATE TABLE IF NOT EXISTS cloudwatch_alert_subscriptions (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id VARCHAR(36) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    alarm_arn VARCHAR(500) NOT NULL,
    notification_webhook_url VARCHAR(500) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
