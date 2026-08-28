variable "cloud_provider" {
  description = "Target cloud provider (aws, gcp, azure)"
  type        = string
  default     = "aws"
}

variable "environment" {
  description = "Deployment environment (production, staging, dev)"
  type        = string
  default     = "production"
}

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC network"
  type        = string
  default     = "10.0.0.0/16"
}

variable "db_instance_class" {
  description = "PostgreSQL RDS instance class"
  type        = string
  default     = "db.r6g.xlarge"
}

variable "redis_node_type" {
  description = "ElastiCache Redis cluster node size"
  type        = string
  default     = "cache.r6g.large"
}

variable "min_worker_nodes" {
  description = "Minimum Kubernetes worker nodes"
  type        = number
  default     = 3
}

variable "max_worker_nodes" {
  description = "Maximum Kubernetes autoscaled worker nodes"
  type        = number
  default     = 20
}
