# ModelForge AI - Google Cloud BigQuery Datalake

resource "google_bigquery_dataset" "modelforge_datalake" {
  dataset_id                  = "modelforge_production_warehouse"
  friendly_name               = "ModelForge Enterprise Datalake"
  description                 = "Central feature store and historical inference telemetry warehouse"
  location                    = var.gcp_region
  default_table_expiration_ms = 315360000000 # 10 years

  labels = {
    env = "production"
  }
}

resource "google_bigquery_table" "feature_snapshots" {
  dataset_id = google_bigquery_dataset.modelforge_datalake.dataset_id
  table_id   = "feature_entity_snapshots"

  time_partitioning {
    type  = "DAY"
    field = "event_timestamp"
  }

  clustering = ["entity_id", "feature_group"]

  schema = jsonencode([
    { name = "entity_id", type = "STRING", mode = "REQUIRED" },
    { name = "event_timestamp", type = "TIMESTAMP", mode = "REQUIRED" },
    { name = "feature_group", type = "STRING", mode = "REQUIRED" },
    { name = "feature_payload", type = "JSON", mode = "REQUIRED" }
  ])
}
