terraform {
  backend "s3" {
    bucket         = "bmin5100-terraform-state"
    key            = "msinger1@sas.upenn.edu-singer-strokepredictor/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
  }
}