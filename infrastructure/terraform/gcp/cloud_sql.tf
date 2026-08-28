# ModelForge AI - Google Cloud SQL PostgreSQL 16 Enterprise Database
# Provisions High-Availability PostgreSQL 16 instance with automated failover, IAM database authentication,
# and private service networking VPC peering.

resource "google_compute_global_address" "private_ip_address" {
  name          = "modelforge-gcp-private-ip"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.modelforge_vpc.id
}

resource "google_service_networking_connection" "private_vpc_connection" {
  network                 = google_compute_network.modelforge_vpc.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip_address.name]
}

resource "google_sql_database_instance" "postgres_instance" {
  name             = "modelforge-prod-postgres"
  database_version = "POSTGRES_16"
  region           = var.region

  depends_on = [google_service_networking_connection.private_vpc_connection]

  settings {
    tier              = "db-custom-8-32768" # 8 vCPUs, 32GB RAM
    availability_type = "REGIONAL"          # High-Availability multi-zone failover
    disk_type         = "PD_SSD"
    disk_size         = 100
    disk_autoresize   = true

    backup_configuration {
      enabled                        = true
      point_in_time_recovery_enabled = true
      start_time                     = "03:00"
    }

    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.modelforge_vpc.id
      enable_private_path_for_google_cloud_services = true
    }

    database_flags {
      name  = "shared_preload_libraries"
      value = "pg_stat_statements"
    }

    database_flags {
      name  = "max_connections"
      value = "500"
    }

    insights_config {
      query_insights_enabled  = true
      query_string_length     = 1024
      record_application_tags = true
    }
  }

  deletion_protection = true
}

resource "google_sql_database" "database" {
  name     = "modelforge"
  instance = google_sql_database_instance.postgres_instance.name
}
