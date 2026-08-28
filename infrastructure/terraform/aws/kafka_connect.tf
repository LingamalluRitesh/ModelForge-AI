# ModelForge AI - AWS MSK Kafka Connect S3 Sink

resource "aws_mskconnect_connector" "s3_feature_sink" {
  name = "modelforge-s3-telemetry-sink"

  kafkaconnect_version = "2.7.1"
  service_execution_role_arn = aws_iam_role.msk_connect_role.arn

  capacity {
    autoscaling {
      min_worker_count = 1
      max_worker_count = 4
      scale_in_policy {
        cpu_utilization_percentage = 20
      }
      scale_out_policy {
        cpu_utilization_percentage = 80
      }
    }
  }

  connector_configuration = {
    "connector.class"                = "io.confluent.connect.s3.S3SinkConnector"
    "tasks.max"                      = "4"
    "topics"                         = "modelforge-live-telemetry"
    "s3.region"                      = var.aws_region
    "s3.bucket.name"                 = aws_s3_bucket.model_artifacts.bucket
    "flush.size"                     = "10000"
    "storage.class"                  = "io.confluent.connect.s3.storage.S3Storage"
    "format.class"                   = "io.confluent.connect.s3.format.parquet.ParquetFormat"
  }

  kafka_cluster {
    apache_kafka_cluster {
      bootstrap_servers = aws_msk_cluster.telemetry_kafka.bootstrap_brokers_tls
      vpc {
        security_groups = [aws_security_group.msk_sg.id]
        subnets         = [aws_subnet.private_subnet_1.id, aws_subnet.private_subnet_2.id]
      }
    }
  }

  kafka_cluster_client_authentication {
    authentication_type = "IAM"
  }

  kafka_cluster_encryption_in_transit {
    encryption_type = "TLS"
  }

  plugin {
    custom_plugin {
      arn      = aws_mskconnect_custom_plugin.s3_sink_plugin.arn
      revision = aws_mskconnect_custom_plugin.s3_sink_plugin.latest_revision
    }
  }
}
