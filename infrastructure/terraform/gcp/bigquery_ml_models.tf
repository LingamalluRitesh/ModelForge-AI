# ModelForge AI - GCP BigQuery ML Model & Feature Export Pipelines

resource "google_bigquery_dataset" "modelforge_feature_store" {
  dataset_id                  = "modelforge_enterprise_features"
  friendly_name               = "ModelForge Enterprise Feature Store"
  description                 = "Materialized feature views for BigQuery ML and Vertex AI training"
  location                    = "US"
  default_table_expiration_ms = 31536000000 # 365 days

  labels = {
    env = "production"
  }
}
