# ModelForge AI - HashiCorp Consul Service Mesh

resource "aws_security_group" "consul_mesh_sg" {
  name        = "modelforge-consul-mesh-sg"
  description = "Security group for Consul Service Mesh mTLS sidecars"
  vpc_id      = aws_vpc.modelforge_vpc.id

  ingress {
    from_port   = 8300
    to_port     = 8302
    protocol    = "tcp"
    self        = true
    description = "Consul server RPC and Serf"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Environment = "production"
  }
}
