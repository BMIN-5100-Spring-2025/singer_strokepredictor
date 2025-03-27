resource "aws_cloudwatch_log_group" "ecs_log_group" {
  name              = "/ecs/singer-strokepredictor"
  retention_in_days = 7

  tags = {
    Name = "singer-strokepredictor-logs"
  }
}