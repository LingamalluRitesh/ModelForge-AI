# ModelForge AI - HashiCorp Vault Enterprise Key Management

resource "aws_kms_key" "vault_unseal_key" {
  description             = "KMS Key for HashiCorp Vault Auto-Unseal"
  deletion_window_in_days = 10
  enable_key_rotation     = true

  tags = {
    Environment = "production"
    Application = "ModelForge-Vault"
  }
}

resource "aws_s3_bucket" "vault_storage" {
  bucket        = "modelforge-vault-backend-${var.aws_account_id}"
  force_destroy = false
}
