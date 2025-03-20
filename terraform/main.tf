data "aws_caller_identity" "current" {}

data "aws_region" "current_region" {}

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