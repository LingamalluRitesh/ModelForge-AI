# ModelForge AI - AWS CloudWatch High-Priority Operational Alarms

resource "aws_cloudwatch_metric_alarm" "p99_latency_breach" {
  alarm_name          = "modelforge-p99-latency-breach"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 2
  metric_name         = "Latency"
  namespace           = "AWS/SageMaker"
  period              = 60
  statistic           = "p99"
  threshold           = 10.0 # 10ms SLA
  alarm_description   = "Inference P99 latency exceeded 10ms SLA limit"
  treat_missing_data  = "notBreaching"

  dimensions = {
    EndpointName = "modelforge-production-endpoint"
  }

  tags = {
    Environment = "production"
    Application = "ModelForge-AI"
  }
}

resource "aws_cloudwatch_metric_alarm" "drift_detected_breach" {
  alarm_name          = "modelforge-population-drift-breach"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 1
  metric_name         = "PopulationStabilityIndex"
  namespace           = "ModelForge/Monitoring"
  period              = 300
  statistic           = "Maximum"
  threshold           = 0.20
  alarm_description   = "PSI statistical drift breached critical threshold 0.20"

  tags = {
    Environment = "production"
  }
}
