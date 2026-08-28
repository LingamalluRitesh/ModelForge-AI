# ModelForge AI - Datadog Synthetic Telemetry Monitors

resource "aws_sns_topic" "modelforge_pagerduty_sns" {
  name = "modelforge-pagerduty-alerts"
}

resource "aws_sns_topic_subscription" "pagerduty_target" {
  topic_arn = aws_sns_topic.modelforge_pagerduty_sns.arn
  protocol  = "https"
  endpoint  = "https://events.pagerduty.com/integration/modelforge/enqueue"
}
