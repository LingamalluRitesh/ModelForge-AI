# ModelForge AI - AWS OpenSearch Distributed Log & Vector Cluster

resource "aws_opensearch_domain" "modelforge_opensearch" {
  domain_name    = "modelforge-enterprise-search"
  engine_version = "OpenSearch_2.11"

  cluster_config {
    instance_type          = "r6g.large.search"
    instance_count         = 3
    dedicated_master_enabled = true
    dedicated_master_type  = "m6g.large.search"
    dedicated_master_count = 3
    zone_awareness_enabled = true
  }

  ebs_options {
    ebs_enabled = true
    volume_type = "gp3"
    volume_size = 250
  }

  encrypt_at_rest {
    enabled = true
  }

  node_to_node_encryption {
    enabled = true
  }

  domain_endpoint_options {
    enforce_https       = true
    tls_security_policy = "Policy-Min-TLS-1-2-2019-07"
  }

  tags = {
    Environment = "production"
    ManagedBy   = "Terraform"
  }
}
