resource "aws_s3_bucket" "singer-strokepredictor" {
  bucket = "singer-strokepredictor"

  tags = {
    Owner = element(split("/", data.aws_caller_identity.current.arn), 1)
  }
}

resource "aws_s3_bucket_ownership_controls" "singer-strokepredictor_ownership_controls" {
  bucket = aws_s3_bucket.singer-strokepredictor.id
  rule {
    object_ownership = "BucketOwnerPreferred"
  }
}

resource "aws_s3_bucket_acl" "singer-strokepredictor_acl" {
  depends_on = [aws_s3_bucket_ownership_controls.singer-strokepredictor_ownership_controls]
  bucket = aws_s3_bucket.singer-strokepredictor.id
  acl    = "private"
}

resource "aws_s3_bucket_lifecycle_configuration" "singer-strokepredictor_expiration" {
  bucket = aws_s3_bucket.singer-strokepredictor.id

  rule {
    id      = "compliance-retention-policy"
    status  = "Enabled"

    expiration {
	  days = 100
    }
  }
}

module "invoke_fargate_lambda" {
  source = "git@github.com:BMIN-5100-Spring-2025/infrastructure.git//invoke_fargate_lambda/terraform?ref=f844e9c04f901768ccb99aff77286165bf71b83e"

  project_name = "singer-strokepredictor"
  ecs_task_definition_arn = aws_ecs_task_definition.singer_task.arn
  ecs_task_execution_role_arn = aws_iam_role.ecs_execution_role.arn
  ecs_task_task_role_arn = aws_iam_role.ecs_task_role.arn
  ecs_task_definition_container_name = "singer-container"

  ecs_cluster_arn = data.terraform_remote_state.infrastructure.outputs.ecs_cluster_arn
  ecs_security_group_id = data.terraform_remote_state.infrastructure.outputs.ecs_security_group_id
  private_subnet_id = data.terraform_remote_state.infrastructure.outputs.private_subnet_id
  api_gateway_authorizer_id = data.terraform_remote_state.infrastructure.outputs.api_gateway_authorizer_id
  api_gateway_execution_arn = data.terraform_remote_state.infrastructure.outputs.api_gateway_execution_arn
  api_gateway_id = data.terraform_remote_state.infrastructure.outputs.api_gateway_id
  environment_variables = {}
}