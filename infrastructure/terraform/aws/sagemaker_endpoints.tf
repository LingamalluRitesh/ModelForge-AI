# ModelForge AI - AWS SageMaker Real-Time Endpoint Infrastructure

resource "aws_sagemaker_model" "modelforge_sagemaker_model" {
  name               = "modelforge-champion-model"
  execution_role_arn = aws_iam_role.sagemaker_execution_role.arn

  primary_container {
    image          = "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.0.0-gpu-py310"
    model_data_url = "s3://${aws_s3_bucket.model_artifacts.bucket}/models/champion/model.tar.gz"
  }
}

resource "aws_sagemaker_endpoint_configuration" "modelforge_endpoint_config" {
  name = "modelforge-champion-endpoint-config"

  production_variants {
    variant_name           = "primary"
    model_name             = aws_sagemaker_model.modelforge_sagemaker_model.name
    initial_instance_count = 2
    instance_type          = "ml.g5.2xlarge"
    initial_variant_weight = 1.0
  }
}
