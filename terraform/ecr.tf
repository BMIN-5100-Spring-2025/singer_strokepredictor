resource "aws_ecr_repository" "singer_strokepredictor" {
  name                 = "singer_strokepredictor"
  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Name        = "singer_strokepredictor"
    Environment = "dev"
  }
}
