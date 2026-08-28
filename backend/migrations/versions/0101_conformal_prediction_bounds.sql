-- ModelForge AI - PostgreSQL Migration 0101: Conformal Prediction Calibration DDL

CREATE TABLE IF NOT EXISTS conformal_prediction_calibration_bounds (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version_id VARCHAR(36) NOT NULL REFERENCES model_versions(id) ON DELETE CASCADE,
    confidence_level FLOAT DEFAULT 0.90,
    q_hat_margin FLOAT NOT NULL,
    calibration_samples_count INT NOT NULL,
    calibrated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
