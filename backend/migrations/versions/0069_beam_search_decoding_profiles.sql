-- ModelForge AI - PostgreSQL Migration 0069: Beam Search Decoding Profiles DDL

CREATE TABLE IF NOT EXISTS beam_search_decoding_profiles (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version_id VARCHAR(36) NOT NULL REFERENCES model_versions(id) ON DELETE CASCADE,
    beam_width INT DEFAULT 5,
    length_penalty_alpha FLOAT DEFAULT 0.6,
    no_repeat_ngram_size INT DEFAULT 3,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
