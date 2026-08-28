# ModelForge AI - S3 DataLake & Model Artifact Storage
# Configures immutable versioning, AES256 server-side encryption, lifecycle transition policies,
# and restrictive IAM bucket policies for bronze/silver/gold feature store and model checkpoints.

resource "aws_s3_bucket" "model_artifacts" {
  bucket        = "modelforge-${var.environment}-model-artifacts-${aws_vpc.modelforge_vpc.id}"
  force_destroy = false

  tags = {
    Environment = var.environment
    Service     = "ModelForgeArtifacts"
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

resource "aws_s3_bucket_public_access_block" "block_public_artifacts" {
  bucket = aws_s3_bucket.model_artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "artifacts_lifecycle" {
  bucket = aws_s3_bucket.model_artifacts.id

  rule {
    id     = "archive-old-experiments"
    status = "Enabled"

    filter {
      prefix = "experiments/archive/"
    }

    transition {
      days          = 90
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 365
      storage_class = "GLACIER"
    }

    noncurrent_version_expiration {
      noncurrent_days = 730
    }
  }
}
