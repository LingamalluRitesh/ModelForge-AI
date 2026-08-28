# ModelForge AI - Google Cloud Platform (GCP) Terraform Module
# Provisions VPC, GKE Autopilot / Standard Cluster, Cloud SQL PostgreSQL, MemoryStore Redis, GCS Bucket, and IAM

variable "environment" { type = string }
variable "project_id" { type = string, default = "modelforge-prod-ai" }
variable "region" { type = string, default = "us-central1" }

# 1. Google Cloud Storage Bucket for Model Artifacts
resource "google_storage_bucket" "model_artifacts" {
  name          = "modelforge-artifacts-${var.environment}-${var.project_id}"
  location      = var.region
  force_destroy = false

  versioning {
    enabled = true
  }

  uniform_bucket_level_access = true
}

# 2. VPC Network & Subnet
resource "google_compute_network" "vpc_network" {
  name                    = "modelforge-${var.environment}-vpc"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "subnet" {
  name          = "modelforge-${var.environment}-subnet"
  ip_cidr_range = "10.10.0.0/16"
  region        = var.region
  network       = google_compute_network.vpc_network.id
}

# 3. Google Kubernetes Engine (GKE) Cluster
resource "google_container_cluster" "primary" {
  name     = "modelforge-${var.environment}-gke"
  location = var.region
  network  = google_compute_network.vpc_network.name
  subnetwork = google_compute_subnetwork.subnet.name

  initial_node_count = 3

  node_config {
    machine_type = "e2-standard-4"
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]
  }
}

# 4. Cloud SQL PostgreSQL Instance
resource "google_sql_database_instance" "postgres" {
  name             = "modelforge-${var.environment}-pg"
  database_version = "POSTGRES_16"
  region           = var.region

  settings {
    tier = "db-custom-4-16384"
    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.vpc_network.id
    }
    backup_configuration {
      enabled = true
    }
  }
}
