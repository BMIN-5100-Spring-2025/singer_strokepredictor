resource "aws_s3_bucket_cors_configuration" "singer_strokepredictor_cors_configuration" {
  bucket = aws_s3_bucket.singer-strokepredictor.id

  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET", "POST", "PUT", "HEAD"]
    allowed_origins = ["http://localhost:*", "bmin5100.com", "*.bmin5100"]
    expose_headers  = ["ETag"]
    max_age_seconds = 3000
  }
} 