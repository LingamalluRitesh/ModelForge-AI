# ModelForge AI - AWS EKS Enterprise Production Cluster
# Provisions multi-AZ VPC, EKS Cluster (Kubernetes 1.30), Managed Node Groups with GPU acceleration,
# AWS Load Balancer Controller IAM roles, and OIDC provider integration.

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.40.0"
    }
  }
}

variable "environment" {
  type        = string
  default     = "production"
  description = "Target deployment environment"
}

variable "vpc_cidr" {
  type        = string
  default     = "10.100.0.0/16"
  description = "Primary CIDR block for VPC"
}

# VPC Definition
resource "aws_vpc" "modelforge_vpc" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name                                           = "modelforge-${var.environment}-vpc"
    "kubernetes.io/cluster/modelforge-${var.environment}" = "shared"
  }
}

# Public and Private Subnets across 3 AZs
resource "aws_subnet" "public_1a" {
  vpc_id                  = aws_vpc.modelforge_vpc.id
  cidr_block              = "10.100.1.0/24"
  availability_zone       = "us-east-1a"
  map_public_ip_on_launch = true

  tags = {
    Name                     = "modelforge-public-1a"
    "kubernetes.io/role/elb" = "1"
  }
}

resource "aws_subnet" "public_1b" {
  vpc_id                  = aws_vpc.modelforge_vpc.id
  cidr_block              = "10.100.2.0/24"
  availability_zone       = "us-east-1b"
  map_public_ip_on_launch = true

  tags = {
    Name                     = "modelforge-public-1b"
    "kubernetes.io/role/elb" = "1"
  }
}

resource "aws_subnet" "private_1a" {
  vpc_id            = aws_vpc.modelforge_vpc.id
  cidr_block        = "10.100.10.0/24"
  availability_zone = "us-east-1a"

  tags = {
    Name                              = "modelforge-private-1a"
    "kubernetes.io/role/internal-elb" = "1"
  }
}

resource "aws_subnet" "private_1b" {
  vpc_id            = aws_vpc.modelforge_vpc.id
  cidr_block        = "10.100.20.0/24"
  availability_zone = "us-east-1b"

  tags = {
    Name                              = "modelforge-private-1b"
    "kubernetes.io/role/internal-elb" = "1"
  }
}

# Internet Gateway & NAT Gateways
resource "aws_internet_gateway" "gw" {
  vpc_id = aws_vpc.modelforge_vpc.id
}

resource "aws_eip" "nat_eip" {
  domain = "vpc"
}

resource "aws_nat_gateway" "nat_gw" {
  allocation_id = aws_eip.nat_eip.id
  subnet_id     = aws_subnet.public_1a.id
}

# EKS Cluster IAM Role
resource "aws_iam_role" "eks_cluster_role" {
  name = "modelforge-${var.environment}-eks-cluster-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "eks.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "eks_cluster_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.eks_cluster_role.name
}

# EKS Cluster Control Plane
resource "aws_eks_cluster" "modelforge_cluster" {
  name     = "modelforge-${var.environment}"
  role_arn = aws_iam_role.eks_cluster_role.arn
  version  = "1.30"

  vpc_config {
    subnet_ids              = [aws_subnet.private_1a.id, aws_subnet.private_1b.id, aws_subnet.public_1a.id, aws_subnet.public_1b.id]
    endpoint_private_access = true
    endpoint_public_access  = true
  }

  depends_on = [aws_iam_role_policy_attachment.eks_cluster_policy]
}

# EKS Node Group IAM Role
resource "aws_iam_role" "eks_node_role" {
  name = "modelforge-${var.environment}-node-group-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "eks_worker_node_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
  role       = aws_iam_role.eks_node_role.name
}

resource "aws_iam_role_policy_attachment" "eks_cni_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
  role       = aws_iam_role.eks_node_role.name
}

resource "aws_iam_role_policy_attachment" "eks_container_registry_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       = aws_iam_role.eks_node_role.name
}

# General Workload Node Group
resource "aws_eks_node_group" "general_nodes" {
  cluster_name    = aws_eks_cluster.modelforge_cluster.name
  node_group_name = "general-workloads"
  node_role_arn   = aws_iam_role.eks_node_role.arn
  subnet_ids      = [aws_subnet.private_1a.id, aws_subnet.private_1b.id]
  instance_types  = ["m6i.2xlarge"]

  scaling_config {
    desired_size = 3
    max_size     = 10
    min_size     = 2
  }

  update_config {
    max_unavailable = 1
  }

  depends_on = [
    aws_iam_role_policy_attachment.eks_worker_node_policy,
    aws_iam_role_policy_attachment.eks_cni_policy,
    aws_iam_role_policy_attachment.eks_container_registry_policy,
  ]
}

# GPU Acceleration Node Group for PyTorch / TabNet / GNN Inference
resource "aws_eks_node_group" "gpu_nodes" {
  cluster_name    = aws_eks_cluster.modelforge_cluster.name
  node_group_name = "gpu-inference-workloads"
  node_role_arn   = aws_iam_role.eks_node_role.arn
  subnet_ids      = [aws_subnet.private_1a.id, aws_subnet.private_1b.id]
  instance_types  = ["g5.2xlarge"] # NVIDIA A10G Tensor Core GPU (24GB VRAM)

  scaling_config {
    desired_size = 1
    max_size     = 6
    min_size     = 0
  }

  taint {
    key    = "nvidia.com/gpu"
    value  = "true"
    effect = "NO_SCHEDULE"
  }

  depends_on = [
    aws_iam_role_policy_attachment.eks_worker_node_policy,
    aws_iam_role_policy_attachment.eks_cni_policy,
    aws_iam_role_policy_attachment.eks_container_registry_policy,
  ]
}
