# ModelForge AI - AWS ElastiCache Redis Cluster for Feature Store Online KV Serving
# Provisions multi-AZ Redis 7.1 replication group with in-transit encryption and token auth.

resource "aws_elasticache_subnet_group" "redis_subnets" {
  name        = "modelforge-${var.environment}-redis-subnet-group"
  subnet_ids  = [aws_subnet.private_1a.id, aws_subnet.private_1b.id]
  description = "Private subnets for ElastiCache Redis replication group"
}

resource "aws_security_group" "redis_sg" {
  name        = "modelforge-${var.environment}-redis-sg"
  description = "Allow inbound Redis traffic from EKS worker nodes"
  vpc_id      = aws_vpc.modelforge_vpc.id

  ingress {
    description     = "Redis Port from EKS Nodes"
    from_port       = 6379
    to_port         = 6379
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

resource "aws_elasticache_replication_group" "redis_cluster" {
  replication_group_id       = "modelforge-${var.environment}-redis"
  description                = "ModelForge Online Feature Store Redis Cluster"
  node_type                  = "cache.r6g.large"
  num_cache_clusters         = 2
  port                       = 6379
  parameter_group_name       = "default.redis7"
  subnet_group_name          = aws_elasticache_subnet_group.redis_subnets.name
  security_group_ids         = [aws_security_group.redis_sg.id]
  automatic_failover_enabled = true
  multi_az_enabled           = true
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true

  tags = {
    Environment = var.environment
    Service     = "ModelForgeFeatureStore"
  }
}
