-- ModelForge AI - PostgreSQL Migration 0022: Genetic Symbolic Feature Formulas DDL

CREATE TABLE IF NOT EXISTS symbolic_feature_formulas (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id VARCHAR(36) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    expression_string TEXT NOT NULL,
    formula_ast JSONB NOT NULL,
    fitness_mse FLOAT NOT NULL,
    generation_discovered INT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
