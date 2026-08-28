# ModelForge AI - AWS EMR Serverless Spark Analytics Engine

resource "aws_emrserverless_application" "modelforge_spark" {
  name          = "modelforge-analytics-spark"
  release_label = "emr-6.15.0"
  type          = "spark"

  initial_capacity {
    initial_capacity_type = "Driver"
    initial_capacity_config {
      worker_count = 2
      worker_configuration {
        cpu    = "4 vCPU"
        memory = "16 GB"
      }
    }
  }

  initial_capacity {
    initial_capacity_type = "Executor"
    initial_capacity_config {
      worker_count = 8
      worker_configuration {
        cpu    = "8 vCPU"
        memory = "32 GB"
      }
    }
  }

  maximum_capacity {
    cpu    = "128 vCPU"
    memory = "512 GB"
  }

  auto_start_configuration {
    enabled = true
  }

  auto_stop_configuration {
    enabled              = true
    idle_timeout_minutes = 15
  }

  tags = {
    Environment = "production"
    Application = "ModelForge-EMR"
  }
}
