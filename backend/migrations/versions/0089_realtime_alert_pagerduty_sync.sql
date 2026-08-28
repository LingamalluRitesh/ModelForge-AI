-- ModelForge AI - PostgreSQL Migration 0089: Realtime Alert PagerDuty Dispatch Sync DDL

CREATE TABLE IF NOT EXISTS alert_pagerduty_dispatch_logs (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    alert_id VARCHAR(36) NOT NULL,
    pagerduty_incident_id VARCHAR(100) NOT NULL,
    incident_status VARCHAR(50) DEFAULT 'TRIGGERED',
    acknowledged_by VARCHAR(150),
    dispatched_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
