# ModelForge AI - AWS Kinesis Streaming Ingestion

resource "aws_kinesis_stream" "telemetry_stream" {
  name             = "modelforge-live-telemetry-stream"
  shard_count      = 4
  retention_period = 48

  shard_level_metrics = [
    "IncomingBytes",
    "IncomingRecords",
    "OutgoingBytes",
    "OutgoingRecords",
    "WriteProvisionedThroughputExceeded",
    "ReadProvisionedThroughputExceeded",
    "IteratorAgeMilliseconds",
  ]

  tags = {
    Environment = "production"
    Application = "ModelForge-AI"
  }
}
