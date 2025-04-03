resource "aws_ecs_task_definition" "singer_task" {
  family                   = "singer-strokepredictor"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "1024"         # 1 vCPU
  memory                   = "2048"         # 2 GB
  execution_role_arn       = aws_iam_role.ecs_execution_role.arn
  task_role_arn            = aws_iam_role.ecs_task_role.arn

  ephemeral_storage {
    size_in_gib = 50
  }

  container_definitions = jsonencode([
    {
      name      = "singer-container"
      image     = "${aws_ecr_repository.singer_strokepredictor.repository_url}:latest"
      cpu       = 1024
      memory    = 2048
      essential = true
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.ecs_log_group.name
          awslogs-region        = "us-east-1"
          awslogs-stream-prefix = "ecs"
        }
      }
      environment = [
        {
          name  = "S3_BUCKET_NAME"
          value = aws_s3_bucket.singer-strokepredictor.id
        },
        {
          name  = "RUN_ENV"
          value = "fargate"
        }
      ]
    }
  ])
}
