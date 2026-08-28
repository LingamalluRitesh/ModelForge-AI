# ModelForge AI - AWS Terraform Module
# Provisions VPC, Private Subnets, EKS Cluster, RDS Aurora PostgreSQL, ElastiCache Redis, S3 Bucket, and IAM

variable "environment" { type = string }
variable "vpc_cidr" { type = string }
variable "eks_cluster_name" { type = string }
variable "db_instance_class" { type = string }
variable "redis_node_type" { type = string }
variable "s3_bucket_name" { type = string }
variable "min_worker_nodes" { type = number }
variable "max_worker_nodes" { type = number }

# 1. S3 Artifacts Storage Bucket with Versioning & Encryption
resource "aws_s3_bucket" "model_artifacts" {
  bucket = var.s3_bucket_name
  tags = {
    Environment = var.environment
    Platform    = "ModelForge"
  }
}

resource "aws_s3_bucket_versioning" "artifacts_versioning" {
  bucket = aws_s3_bucket.model_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts_encryption" {
  bucket = aws_s3_bucket.model_artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# 2. VPC & Subnets
resource "aws_vpc" "modelforge_vpc" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true
  tags = {
    Name        = "modelforge-${var.environment}-vpc"
    Environment = var.environment
  }
}

resource "aws_subnet" "public_1" {
  vpc_id                  = aws_vpc.modelforge_vpc.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "us-east-1a"
  map_public_ip_on_launch = true
  tags = { Name = "modelforge-public-1" }
}

resource "aws_subnet" "private_1" {
  vpc_id            = aws_vpc.modelforge_vpc.id
  cidr_block        = "10.0.10.0/24"
  availability_zone = "us-east-1a"
  tags = { Name = "modelforge-private-1" }
}

# 3. RDS PostgreSQL Instance
resource "aws_db_instance" "postgres" {
  identifier             = "modelforge-${var.environment}-db"
  allocated_storage      = 100
  max_allocated_storage  = 1000
  engine                 = "postgres"
  engine_version         = "16.1"
  instance_class         = var.db_instance_class
  db_name                = "modelforge_db"
  username               = "modelforge_admin"
  password               = "ChangeMeInProductionVault123!"
  skip_final_snapshot    = true
  multi_az               = true
  publicly_accessible    = false
  vpc_security_group_ids = [aws_security_group.db_sg.id]
}

resource "aws_security_group" "db_sg" {
  name        = "modelforge-db-sg"
  description = "Allow inbound PostgreSQL traffic from EKS"
  vpc_id      = aws_vpc.modelforge_vpc.id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
