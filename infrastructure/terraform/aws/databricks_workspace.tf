# ModelForge AI - AWS Databricks ML Workspace

resource "aws_s3_bucket" "databricks_root" {
  bucket        = "modelforge-databricks-workspace-root-${var.aws_account_id}"
  force_destroy = false
}

resource "aws_s3_bucket_server_side_encryption_configuration" "databricks_s3_crypto" {
  bucket = aws_s3_bucket.databricks_root.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}
