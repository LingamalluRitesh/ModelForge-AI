-- ModelForge AI - PostgreSQL Migration 0027: Lottery Ticket Pruning Masks DDL

CREATE TABLE IF NOT EXISTS lottery_ticket_pruning_masks (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version_id VARCHAR(36) NOT NULL REFERENCES model_versions(id) ON DELETE CASCADE,
    sparsity_ratio FLOAT NOT NULL,
    pruning_rounds INT DEFAULT 5,
    storage_mask_uri VARCHAR(500) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
