# Configure the Google Cloud provider
provider "google" {
  project = "cogent-theater-294521" # Your Project ID
  region  = "us-central1"
}

# Enable required APIs - Terraform will enable these if they aren't already
resource "google_project_service" "cloud_run_api" {
  service            = "run.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "cloud_storage_api" {
  service            = "storage.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "vertex_ai_api" {
  service            = "aiplatform.googleapis.com"
  disable_on_destroy = false
}

resource "google_cloud_run_v2_service" "rag_app_cloud_run" {
  name     = "rag-app-cloud-run"
  location = "us-central1"
  # Ensures the Cloud Run API is enabled before trying to create the service
  depends_on = [
    google_project_service.cloud_run_api
  ]

  template {
    containers {
      image = "us-docker.pkg.dev/cloudrun/container/hello" # Placeholder image
      ports {
        container_port = 8080
      }
    }
    # This is the correct way to specify the service account for Cloud Run V2.
    # Replace <PROJECT_NUMBER> with your actual project number (168130626980)
    service_account = "168130626980-compute@developer.gserviceaccount.com"
  }

  # Allow unauthenticated access for demo purposes. Remove or restrict in production.
  ingress = "INGRESS_TRAFFIC_ALL"
}

# Grant all users the ability to invoke the Cloud Run service.
# This is for testing. For production, define specific principals.
# resource "google_cloud_run_service_iam_member" "rag_app_cloud_run_invoker" {
#   location = google_cloud_run_v2_service.rag_app_cloud_run.location
#   project  = google_cloud_run_v2_service.rag_app_cloud_run.project
#   service  = google_cloud_run_v2_service.rag_app_cloud_run.name
#   role     = "roles/run.invoker"
#   member   = "allUsers" # Grants public access, be cautious in production
#   depends_on = [
#     google_cloud_run_v2_service.rag_app_cloud_run
#   ]
# }

# my access only, must use iam binding instead of iam member, the latter does NOT remove prior authorization.
resource "google_cloud_run_service_iam_binding" "rag_app_cloud_run_invoker" {
  location = google_cloud_run_v2_service.rag_app_cloud_run.location
  project  = google_cloud_run_v2_service.rag_app_cloud_run.project
  service  = google_cloud_run_v2_service.rag_app_cloud_run.name
  role     = "roles/run.invoker"
  members = [
    "user:dionsongus@gmail.com", # <-- This makes it private to you
  ]
  depends_on = [
    google_cloud_run_v2_service.rag_app_cloud_run
  ]
}


resource "google_storage_bucket" "rag_app_gcs_bucket" {
  name     = "cogent-theater-294521-rag-app-gcs-bucket" # Bucket names must be globally unique!
  location = "US"
  # Ensures the Cloud Storage API is enabled before trying to create the bucket
  depends_on = [
    google_project_service.cloud_storage_api
  ]
  # Allows the bucket to be destroyed even if it contains objects (useful for testing)
  force_destroy = true
  # Optional: Add uniform bucket-level access for simpler permissions management
  uniform_bucket_level_access = true
  # Optional: Add versioning to protect against accidental deletions
  versioning {
    enabled = true
  }
}

