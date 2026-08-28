# ModelForge AI - AWS RDS PostgreSQL Production Database
# Provisions Multi-AZ PostgreSQL 16 Instance with automated snapshots, KMS encryption at rest,
# and enhanced performance insights for model metadata, audits, and metrics storage.

resource "aws_db_subnet_group" "rds_subnets" {
  name        = "modelforge-${var.environment}-rds-subnet-group"
  subnet_ids  = [aws_subnet.private_1a.id, aws_subnet.private_1b.id]
  description = "Private subnets for RDS PostgreSQL cluster"
}

resource "aws_security_group" "rds_sg" {
  name        = "modelforge-${var.environment}-rds-sg"
  description = "Allow inbound PostgreSQL traffic from EKS worker nodes"
  vpc_id      = aws_vpc.modelforge_vpc.id

  ingress {
    description     = "PostgreSQL from EKS Nodes"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_eks_cluster.modelforge_cluster.vpc_config[0].cluster_security_group_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_db_parameter_group" "pg16_params" {
  name   = "modelforge-${var.environment}-pg16"
  family = "postgres16"

  parameter {
    name  = "shared_preload_libraries"
    value = "pg_stat_statements,pgcrypto"
  }

  parameter {
    name  = "max_connections"
    value = "500"
  }

  parameter {
    name  = "work_mem"
    value = "32768" # 32MB
  }
}

resource "aws_db_instance" "modelforge_postgres" {
  identifier                  = "modelforge-${var.environment}-postgres"
  engine                      = "postgres"
  engine_version              = "16.2"
  instance_class              = "db.r6g.xlarge"
  allocated_storage           = 100
  max_allocated_storage       = 1000
  storage_type                = "gp3"
  storage_encrypted           = true
  multi_az                    = true
  publicly_accessible        = false
  db_subnet_group_name        = aws_db_subnet_group.rds_subnets.name
  vpc_security_group_ids      = [aws_security_group.rds_sg.id]
  parameter_group_name        = aws_db_parameter_group.pg16_params.name
  auto_minor_version_upgrade  = true
  backup_retention_period     = 30
  backup_window               = "03:00-04:00"
  maintenance_window          = "sun:04:30-sun:05:30"
  deletion_protection         = true
  skip_final_snapshot         = false
  final_snapshot_identifier   = "modelforge-${var.environment}-final-snap"
  performance_insights_enabled = true

  tags = {
    Environment = var.environment
    Service     = "ModelForge"
  }
}
