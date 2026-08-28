# ModelForge AI - Terraform Multi-Cloud Infrastructure as Code
# Deploys High-Availability Kubernetes Cluster, PostgreSQL RDS, Redis, S3/GCS/Blob, and IAM Roles.

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

# AWS Module Deployment (Default)
module "aws_infrastructure" {
  count  = var.cloud_provider == "aws" ? 1 : 0
  source = "./modules/aws"

  environment           = var.environment
  vpc_cidr              = var.vpc_cidr
  eks_cluster_name      = "modelforge-${var.environment}-eks"
  db_instance_class     = var.db_instance_class
  redis_node_type       = var.redis_node_type
  s3_bucket_name        = "modelforge-artifacts-${var.environment}-${var.aws_region}"
  min_worker_nodes      = var.min_worker_nodes
  max_worker_nodes      = var.max_worker_nodes
}
