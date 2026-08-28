# ModelForge AI - Google Cloud GKE Enterprise Production Cluster
# Provisions Custom VPC Subnetworks, Google Kubernetes Engine (GKE Autopilot/Standard),
# NVIDIA L4 / A100 GPU Acceleration Node Pools, and Workload Identity Federation.

terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.20.0"
    }
  }
}

variable "project_id" {
  type        = string
  default     = "modelforge-enterprise-prod"
  description = "GCP Project ID"
}

variable "region" {
  type        = string
  default     = "us-central1"
  description = "Primary GCP Region"
}

# GCP VPC Network
resource "google_compute_network" "modelforge_vpc" {
  name                    = "modelforge-prod-vpc"
  auto_create_subnetworks = false
}

# GKE Primary Subnetwork
resource "google_compute_subnetwork" "gke_subnet" {
  name          = "modelforge-gke-subnet"
  ip_cidr_range = "10.200.0.0/20"
  region        = var.region
  network       = google_compute_network.modelforge_vpc.id

  secondary_ip_range {
    range_name    = "gke-pods"
    ip_cidr_range = "10.201.0.0/16"
  }

  secondary_ip_range {
    range_name    = "gke-services"
    ip_cidr_range = "10.202.0.0/20"
  }
}

# Cloud Router & NAT for Private GKE Egress
resource "google_compute_router" "router" {
  name    = "modelforge-gcp-router"
  region  = var.region
  network = google_compute_network.modelforge_vpc.id
}

resource "google_compute_router_nat" "nat" {
  name                               = "modelforge-gcp-nat"
  router                             = google_compute_router.router.name
  region                             = var.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
}

# GKE Service Account
resource "google_service_account" "gke_sa" {
  account_id   = "modelforge-gke-node-sa"
  display_name = "ModelForge GKE Node Service Account"
}

resource "google_project_iam_member" "gke_node_log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.gke_sa.email}"
}

resource "google_project_iam_member" "gke_node_metric_writer" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.gke_sa.email}"
}

# GKE Cluster Control Plane
resource "google_container_cluster" "primary_cluster" {
  name     = "modelforge-production-gke"
  location = var.region

  network    = google_compute_network.modelforge_vpc.name
  subnetwork = google_compute_subnetwork.gke_subnet.name

  remove_default_node_pool = true
  initial_node_count       = 1

  ip_allocation_policy {
    cluster_secondary_range_name  = "gke-pods"
    services_secondary_range_name = "gke-services"
  }

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  release_channel {
    channel = "REGULAR"
  }

  addons_config {
    http_load_balancing {
      disabled = false
    }
    horizontal_pod_autoscaling {
      disabled = false
    }
    gce_persistent_disk_csi_driver_config {
      enabled = true
    }
  }
}

# General Purpose Node Pool
resource "google_container_node_pool" "general_nodes" {
  name       = "general-pool"
  location   = var.region
  cluster    = google_container_cluster.primary_cluster.name
  node_count = 3

  autoscaling {
    min_node_count = 3
    max_node_count = 15
  }

  node_config {
    machine_type    = "e2-standard-8"
    service_account = google_service_account.gke_sa.email
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]

    labels = {
      role = "general-serving"
    }
  }
}

# GPU Accelerated Node Pool (NVIDIA L4 Tensor Core GPUs)
resource "google_container_node_pool" "gpu_nodes" {
  name       = "gpu-inference-pool"
  location   = var.region
  cluster    = google_container_cluster.primary_cluster.name
  node_count = 1

  autoscaling {
    min_node_count = 0
    max_node_count = 8
  }

  node_config {
    machine_type    = "g2-standard-8"
    service_account = google_service_account.gke_sa.email
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]

    guest_accelerator {
      type  = "nvidia-l4"
      count = 1
      gpu_driver_installation_config {
        gpu_driver_version = "LATEST"
      }
    }

    taint {
      key    = "nvidia.com/gpu"
      value  = "present"
      effect = "NO_SCHEDULE"
    }

    labels = {
      role = "deep-learning-inference"
    }
  }
}
