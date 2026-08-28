# ModelForge AI - AWS Distributed GPU Worker Pool 3

resource "aws_eks_node_group" "gpu_worker_pool_3" {
  cluster_name    = aws_eks_cluster.modelforge_eks.name
  node_group_name = "modelforge-gpu-pool-3"
  node_role_arn   = aws_iam_role.eks_node_role.arn
  subnet_ids      = [aws_subnet.private_subnet_1.id, aws_subnet.private_subnet_2.id]

  scaling_config {
    desired_size = 2
    max_size     = 8
    min_size     = 1
  }

  instance_types = ["g5.2xlarge"]
  capacity_type  = "ON_DEMAND"

  labels = {
    "workload" = "gpu-training"
    "pool_id"  = "3"
  }

  tags = {
    Environment = "production"
    ManagedBy   = "Terraform"
  }
}
