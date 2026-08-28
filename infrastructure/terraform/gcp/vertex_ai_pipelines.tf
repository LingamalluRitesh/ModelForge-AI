# ModelForge AI - Google Cloud Vertex AI Pipeline Infrastructure

resource "google_vertex_ai_endpoint" "modelforge_vertex_endpoint" {
  name         = "modelforge-production-endpoint"
  display_name = "ModelForge Production Inference Endpoint"
  location     = var.gcp_region
  project      = var.gcp_project_id

  labels = {
    environment = "production"
    managed_by  = "modelforge-ai"
  }
}

resource "google_vertex_ai_featurestore" "modelforge_featurestore" {
  name     = "modelforge_enterprise_featurestore"
  location = var.gcp_region
  project  = var.gcp_project_id

  online_serving_config {
    fixed_node_count = 3
  }
}
